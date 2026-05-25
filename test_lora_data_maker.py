# -------- FEWSHOT PUBLIC BASELINE RUN --------

MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results/best_fewshot_public.jsonl"

import json, os, re, sys
from pathlib import Path
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem carefully and briefly. "
    "Use the <think> section for working, then give the final answer only inside a single \\boxed{}. "
    "The boxed answer must be the last line and must contain no extra words. "
    "If the problem has multiple sub-answers, separate them by commas inside one \\boxed{}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Use the <think> section for a short justification, then output only the chosen letter inside a single \\boxed{}. "
    "The boxed answer must be the last line and must contain no extra words."
)

FEWSHOT_MATH = """
Here are solved examples of the required answer style.

Example 1
Problem: Compute 6 * 7.
<think>
6 * 7 = 42.
</think>
\\boxed{42}

Example 2
Problem: Simplify 12/16.
<think>
Divide numerator and denominator by 4 to get 3/4.
</think>
\\boxed{\\frac{3}{4}}

Now solve the next problem in the same format.
"""

FEWSHOT_MCQ = """
Here are solved examples of the required answer style.

Example 1
Problem: Which option equals 9 - 4?
Options:
A. 3
B. 5
C. 7
<think>
9 - 4 = 5, so option B is correct.
</think>
\\boxed{B}

Example 2
Problem: Which option is the next number after 8?
Options:
A. 7
B. 9
C. 10
<think>
The next integer after 8 is 9, so option B is correct.
</think>
\\boxed{B}

Now solve the next problem in the same format.
"""

def build_few_shot_block(options):
    return FEWSHOT_MCQ if options else FEWSHOT_MATH

def build_prompt(question, options):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options)
        )
        user_prompt = (
            f"{build_few_shot_block(options)}\n\n"
            f"Question:\n{question}\n\nOptions:\n{opts_text}"
        )
        return SYSTEM_PROMPT_MCQ, user_prompt

    user_prompt = f"{build_few_shot_block(options)}\n\nQuestion:\n{question}"
    return SYSTEM_PROMPT_MATH, user_prompt

# ---------------- LOAD DATA ----------------

data = [json.loads(line) for line in open(DATA_PATH)]

# ---------------- MODEL ----------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    max_num_seqs=1,
)

MAX_TOKENS = 8192

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
)

# ---------------- SAVE SETUP ----------------

SAVE_EVAL = True
BATCH_SIZE = 5

out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

count_path = out_path.with_suffix(".count.txt")

if count_path.exists():
    start_idx = int(count_path.read_text().strip())
else:
    start_idx = 0

print(f"Starting from index {start_idx}")

# ---------------- SCORING HELPERS ----------------

def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()

sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)

# ---------------- MAIN LOOP ----------------

pending_results = []

for idx in tqdm(range(start_idx, 200)):

    item = data[idx]

    system, user = build_prompt(item["question"], item.get("options"))

    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate(
        [prompt_text],
        sampling_params=sampling_params,
    )

    response = outputs[0].outputs[0].text.strip()

    # ---------- SCORE ----------

    is_mcq = bool(item.get("options"))
    gold = item["answer"]

    if is_mcq:
        correct = score_mcq(response, str(gold))
    else:
        gold_list = gold if isinstance(gold, list) else [gold]

        try:
            correct = judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        except Exception:
            correct = False

    r = {
        "id": item.get("id"),
        "is_mcq": is_mcq,
        "gold": gold,
        "response": response,
        "correct": correct,
    }

    pending_results.append(r)

    # ---------- SAVE ----------

    if len(pending_results) == BATCH_SIZE or idx == len(data) - 1:

        with open(out_path, "a") as f:
            for r in pending_results:
                f.write(json.dumps(r) + "\n")

        saved_count = idx + 1
        count_path.write_text(str(saved_count))

        print(f"Saved through item {saved_count}")

        pending_results = []

# ---------------- FINAL SCORE ----------------

results = [json.loads(line) for line in open(OUTPUT_PATH)]

mcq_res  = [r for r in results if r["is_mcq"]]
free_res = [r for r in results if not r["is_mcq"]]

def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"MCQ       : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d} ({acc(mcq_res):.2f}%)")
print(f"Free-form : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d} ({acc(free_res):.2f}%)")
print(f"Overall   : {sum(r['correct'] for r in results):4d} / {len(results):4d} ({acc(results):.2f}%)")
print("=" * 50)