import numpy
print(numpy.__version__)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

print("imports work")

# Reflection-pass test runner.
# Goal: run on the public set for only 20 samples.

import json, os, re, sys
from pathlib import Path
from typing import Optional

MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results/reflection_public_20.jsonl"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options") for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

mcq_sample  = next(d for d in data if d.get("options"))
free_sample = next(d for d in data if not d.get("options"))

print("\n── MCQ sample ──")
print(json.dumps(mcq_sample, indent=2))
print("\n── Free-form sample ──")
print(json.dumps(free_sample, indent=2))


SYSTEM_PROMPT_MATH_STAGE1 = (
    "You are an expert mathematician. Solve the problem carefully. "
    "Show brief reasoning in <think> tags, then give a draft answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ_STAGE1 = (
    "You are an expert mathematician. Read the question and options carefully, "
    "solve it, and give a draft answer inside a single \\boxed{} with the chosen letter. "
    "Use short reasoning in <think> tags."
)

SYSTEM_PROMPT_MATH_STAGE2 = (
    "You are checking a prior solution to a math problem. "
    "Read the draft answer, verify the calculations, fix any mistakes, and output only the final cleaned answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ_STAGE2 = (
    "You are checking a prior solution to a multiple-choice problem. "
    "Read the draft answer, verify the option choice, fix any mistakes, and output only the final chosen letter inside one \\boxed{}. "
    "Do not add extra words."
)

FEWSHOT_MATH_STAGE1 = """Here are solved examples of the required answer style.

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

FEWSHOT_MCQ_STAGE1 = """Here are solved examples of the required answer style.

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


FEWSHOT_MATH_STAGE2 = """You are reviewing a draft solution.

Given a problem and a draft answer, check the work carefully, correct any mistake, and return only the final cleaned answer inside \\boxed{}.

Problem: Compute 6 * 7.
Draft answer: 42
<think>
The draft is correct after checking the arithmetic.
</think>
\\boxed{42}

Problem: Simplify 12/16.
Draft answer: 12/16 = 0.75
<think>
The draft is not in the cleanest final form. 12/16 simplifies to 3/4.
</think>
\\boxed{\\frac{3}{4}}
"""

FEWSHOT_MCQ_STAGE2 = """You are reviewing a draft solution.

Given a multiple-choice problem and a draft answer, verify the choice and return only the final letter inside \\boxed{}.

Problem: Which option equals 9 - 4?
Options:
A. 3
B. 5
C. 7
Draft answer: B
<think>
The draft is correct. 9 - 4 = 5, which is option B.
</think>
\\boxed{B}

Problem: Which option is the next number after 8?
Options:
A. 7
B. 9
C. 10
Draft answer: C
<think>
The draft is wrong. The next integer after 8 is 9, which is option B.
</think>
\\boxed{B}
"""


def build_few_shot_block(options: Optional[list], stage: int) -> str:
    if stage == 1:
        return FEWSHOT_MCQ_STAGE1 if options else FEWSHOT_MATH_STAGE1
    return FEWSHOT_MCQ_STAGE2 if options else FEWSHOT_MATH_STAGE2


def build_prompt_stage1(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return the draft-generation prompt for a question."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_prompt = (
            f"{build_few_shot_block(options, 1)}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{opts_text}\n\n"
            "Give a draft answer and keep it inside one boxed letter."
        )
        return SYSTEM_PROMPT_MCQ_STAGE1, user_prompt
    user_prompt = (
        f"{build_few_shot_block(options, 1)}\n\n"
        f"Question:\n{question}\n\n"
        "Give a draft answer and keep it inside one boxed final answer."
    )
    return SYSTEM_PROMPT_MATH_STAGE1, user_prompt


def build_prompt_stage2(question: str, options: Optional[list], draft_response: str) -> tuple[str, str]:
    """Return the reflection prompt that checks and corrects a draft."""
    draft_response = draft_response.strip()
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_prompt = (
            f"{build_few_shot_block(options, 2)}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{opts_text}\n\n"
            f"Draft answer:\n{draft_response}\n\n"
            "Check the draft carefully, correct any mistake, and output only the final letter inside \\boxed{}."
        )
        return SYSTEM_PROMPT_MCQ_STAGE2, user_prompt

    user_prompt = (
        f"{build_few_shot_block(options, 2)}\n\n"
        f"Question:\n{question}\n\n"
        f"Draft answer:\n{draft_response}\n\n"
        "Check the draft carefully, correct any mistake, and output only the final cleaned answer inside \\boxed{}."
    )
    return SYSTEM_PROMPT_MATH_STAGE2, user_prompt


for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
    sys_p, usr_p = build_prompt_stage1(item["question"], item.get("options"))
    print(f"── {label} stage1 user prompt (first 200 chars) ──")
    print(usr_p[:200], "...\n")


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

MAX_TOKENS = 16384

llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    gpu_memory_utilization=0.9,
    max_model_len=24576,
    trust_remote_code=True,
    max_num_seqs=1,
    max_num_batched_tokens=4096,
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
)

reflection_sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.2,
    top_p=0.9,
    top_k=20,
    repetition_penalty=1.0,
)


SAVE_EVAL = True
BATCH_SIZE = 5

out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

count_path = out_path.parent / f"{out_path.stem}_count.txt"

if count_path.exists():
    start_idx = int(count_path.read_text().strip())
else:
    start_idx = 0

run_end_idx = min(20, len(data))
print(f"Starting reflection run at index {start_idx}; ending before {run_end_idx}", flush=True)


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


pending_results = []

if start_idx >= run_end_idx:
    print("Run limit reached for this resume point; nothing to process.", flush=True)

for idx in tqdm(range(start_idx, run_end_idx), desc="Generating", unit="item"):
    item = data[idx]
    item_num = idx - start_idx + 1
    total_items = max(run_end_idx - start_idx, 0)
    print(
        f"[{item_num}/{total_items}] Processing id={item.get('id')} "
        f"({'MCQ' if item.get('options') else 'free-form'})",
        flush=True,
    )

    system1, user1 = build_prompt_stage1(item["question"], item.get("options"))
    prompt_text_1 = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system1},
            {"role": "user", "content": user1},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    output1 = llm.generate([prompt_text_1], sampling_params=sampling_params)
    draft_response = output1[0].outputs[0].text.strip()
    print(f"[{item_num}/{total_items}] Draft preview: {draft_response[:120].replace(chr(10), ' ')}", flush=True)

    system2, user2 = build_prompt_stage2(item["question"], item.get("options"), draft_response)
    prompt_text_2 = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system2},
            {"role": "user", "content": user2},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    output2 = llm.generate([prompt_text_2], sampling_params=reflection_sampling_params)
    response = output2[0].outputs[0].text.strip()
    print(f"[{item_num}/{total_items}] Final preview: {response[:120].replace(chr(10), ' ')}", flush=True)

    is_mcq = bool(item.get("options"))
    gold = item.get("answer")

    r = {
        "id": item.get("id"),
        "is_mcq": is_mcq,
        "draft_response": draft_response,
        "response": response,
    }

    if SAVE_EVAL:
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

        r["gold"] = gold
        r["correct"] = correct

    pending_results.append(r)

    if len(pending_results) == BATCH_SIZE or idx == run_end_idx - 1:
        with open(out_path, "a", encoding="utf-8") as f:
            for r in pending_results:
                if SAVE_EVAL:
                    record = {
                        "id": r["id"],
                        "is_mcq": r["is_mcq"],
                        "gold": r["gold"],
                        "draft_response": r["draft_response"],
                        "response": r["response"],
                        "correct": r["correct"],
                    }
                else:
                    record = {
                        "id": r["id"],
                        "is_mcq": r["is_mcq"],
                        "draft_response": r["draft_response"],
                        "response": r["response"],
                    }

                f.write(json.dumps(record) + "\n")

        saved_count = idx + 1
        count_path.write_text(str(saved_count))

        print(f"Saved through item {saved_count}; batch size {len(pending_results)}", flush=True)
        pending_results = []


results = [json.loads(line) for line in open(OUTPUT_PATH, encoding="utf-8")]
print(f"Scoring complete. {len(results)} results.", flush=True)

mcq_res = [r for r in results if r["is_mcq"]]
free_res = [r for r in results if not r["is_mcq"]]


def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0


print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
print("=" * 50)
