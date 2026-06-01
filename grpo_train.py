"""
GRPO training harness (LoRA / QLoRA) — prototype

This script implements a simplified GRPO training loop using grouped sampling
and LoRA adapters. It is a prototype and intended to run on a machine with
sufficient GPU memory (DSMLP A30 recommended). It does not include advanced
features such as distributed training, checkpointing, or resume logic — add as
needed for long runs.

USAGE (example, run on DSMLP):
    pip install -r requirements.txt
    python grpo_train.py

NOTE: This will modify LoRA weights (small) and requires `bitsandbytes`,
`peft`, and `accelerate`. Only run on the target environment.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from tqdm import tqdm

# Config — tune these for your environment
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "data/public.jsonl"
OUTPUT_PATH = "results/grpo_train_run.jsonl"
GPU_DEVICE = 0

NUM_SAMPLES = 3
MAX_NEW_TOKENS = 512
LR = 1e-4
EPOCHS = 1
BATCH_SIZE = 1  # per update (we generate per example grouped samples)
RUN_LIMIT = 20

# LoRA config
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def build_prompt(question: str, options: Optional[list]) -> str:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_prompt = f"Read the problem and options, then answer.\n\nQuestion:\n{question}\n\nOptions:\n{opts_text}"
        return user_prompt
    return f"Solve the problem and output the final answer in LaTeX or arithmetic.\n\nQuestion:\n{question}"


def extract_boxed_answer(text: str) -> str:
    # Rudimentary extraction: prefer last \boxed{...}
    m = re.search(r"\\boxed\{([^}]*)\}", text[::-1])
    if m:
        # reversed search hack; fallback simple
        pass
    # fallback simple heuristics
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
    return nums[-1] if nums else text.strip()


def score_answer(response: str, gold) -> float:
    # Simple deterministic reward: 1.0 for exact match (after normalize), 0 else.
    pred = extract_boxed_answer(response)
    if isinstance(gold, list):
        gold_norm = [str(x).strip() for x in gold]
        return 1.0 if pred in gold_norm else 0.0
    return 1.0 if pred == str(gold).strip() else 0.0


def sequence_logprob(model, input_ids: torch.Tensor, device: torch.device, gen_start: int) -> float:
    # compute sum logprob for tokens after gen_start
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        logp = F.log_softmax(shift_logits, dim=-1)
        token_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        # mask out tokens before gen_start in labels (since those correspond to prompt)
        mask = torch.arange(shift_labels.size(1), device=device).unsqueeze(0) >= gen_start
        seq_logp = token_logp.masked_select(mask).sum().item()
        return seq_logp


def main():
    device = torch.device(f"cuda:{GPU_DEVICE}" if torch.cuda.is_available() else "cpu")

    data = [json.loads(line) for line in open(DATA_PATH)]
    run_end = min(len(data), RUN_LIMIT)

    # Load tokenizer + model in 4-bit (QLoRA) using bitsandbytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.train()

    # optimizer over LoRA params
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for idx in tqdm(range(0, run_end), desc="GRPO Train"):
            item = data[idx]
            prompt = build_prompt(item["question"], item.get("options"))
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            gen_outputs = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=NUM_SAMPLES,
                return_dict_in_generate=True,
                output_scores=False,
            )

            # gen_outputs.sequences shape: (NUM_SAMPLES, seq_len)
            seqs = gen_outputs.sequences
            # compute logprobs for each generated sequence by re-running model on full sequence
            seq_logps = []
            rewards = []
            device = next(model.parameters()).device
            for s in seqs:
                s = s.unsqueeze(0).to(device)
                gen_start = input_ids.shape[-1]
                lp = sequence_logprob(model, s, device, gen_start)
                seq_logps.append(lp)
                # reconstruct text from generated part
                gen_text = tokenizer.decode(s[0, gen_start:], skip_special_tokens=True)
                r = score_answer(gen_text, item["answer"]) if item.get("answer") is not None else 0.0
                rewards.append(r)

            # compute group baseline and advantages
            mean_r = sum(rewards) / len(rewards)
            advantages = [r - mean_r for r in rewards]

            # policy gradient loss: - sum (adv * logp)
            logps = torch.tensor(seq_logps, dtype=torch.float32, device=device)
            advs = torch.tensor(advantages, dtype=torch.float32, device=device)
            loss = -(advs * logps).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # save diagnostic record
            record = {
                "id": item.get("id"),
                "rewards": rewards,
                "advantages": [float(a) for a in advantages],
                "logps": [float(lp) for lp in seq_logps],
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    # save LoRA adapters
    print("Saving LoRA adapters to results/lora_adapters")
    model.save_pretrained("results/lora_adapters")


if __name__ == '__main__':
    main()
