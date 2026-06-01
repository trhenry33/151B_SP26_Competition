import numpy
print(numpy.__version__)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

print("imports work")


#imports
#baseline is 60% with first 10
#basline is 50% with first 20 - MC: 4/9 FR: 6/11
#reflection is same as baseline
#self consistency is same
#chain of thought is same
#fewshot with others : 55%
#jsut fewshot: 65% - entire: 57.82%

#to test: rm results/count.txt and change the output path on this file to start a new file to check.

#travis test path so far: baseline -> reflective _> MC self consistency _> chain of thought -> fewshot example
#new travis path: baseline -> fewshot -> SFT/LoRA

import json, os, re, sys
from pathlib import Path
from typing import Optional

MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/fewshot_examples_private_fullrun.jsonl"


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

#data



data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

# Preview one MCQ and one free-form item
mcq_sample  = next(d for d in data if d.get("options"))
free_sample = next(d for d in data if not d.get("options"))

print("\n── MCQ sample ──")
print(json.dumps(mcq_sample, indent=2))
print("\n── Free-form sample ──")
print(json.dumps(free_sample, indent=2))

#prompt

# SYSTEM_PROMPT_MATH = (
#     "You are an expert mathematician. Solve the problem step-by-step. "
#     "Think through the problem step-by-step before answering. "
#     "Show your reasoning clearly and logically. "
#     "Before finalizing, verify calculations, algebra, and formatting mistakes. Correct any errors before giving the final boxed answer."
#     "Put your final answer inside \\boxed{}. "
#     "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
#     "e.g. \\boxed{3, 7}."
# )

# SYSTEM_PROMPT_MCQ = (
#     "You are an expert mathematician. "
#     "Think through the problem step-by-step before answering. "
#     "Show your reasoning clearly and logically. "
#     "Solve the problem carefully and consider multiple possible solution paths before choosing an answer. "
#     "Before finalizing, verify calculations, algebra, and formatting mistakes. Correct any errors before giving the final boxed answer." 
#     "Verify the final choice against all options. "
#     "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
# )

# SYSTEM_PROMPT_MATH = (
#     "You are an expert mathematician. Solve carefully but keep the work concise. "
#     "Do NOT write a second explanation after the thinking section. "
#     "Never use \\boxed{} for intermediate values. Use \\boxed{} exactly once, only on the final line. "
#     "If the problem has multiple [ANS] blanks, count them and give answers in the same order, separated by commas. "
#     "If you are unsure or the problem is long, still make your best final answer instead of continuing indefinitely. "
#     "The last line must be exactly one boxed answer and nothing else, e.g. \\boxed{3, 7}."
# )

# SYSTEM_PROMPT_MCQ = (
#     "You are an expert mathematician. Solve carefully but keep the work concise. "
#     "Compare your result against the answer choices before finalizing. "
#     "Do NOT write a second explanation after the thinking section. "
#     "Never use \\boxed{} for intermediate values. Use \\boxed{} exactly once, only on the final line. "
#     "For a normal multiple-choice problem, output one letter only. "
#     "If the problem explicitly asks for multiple choices or multiple blanks, output the letters in order separated by commas. "
#     "If you are unsure, choose the best matching option instead of continuing indefinitely. "
#     "The last line must be exactly one boxed answer and nothing else, e.g. \\boxed{C}."
# )

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician taking a timed exam. Solve directly and carefully. "
    "Do not ramble, debate interpretations, or write a second solution after finishing. "
    "Preserve the requested answer form. Prefer exact symbolic answers when natural, such as fractions, radicals, powers, or pi expressions. "
    "For numerical answers, keep high precision: use at least 8 significant digits when possible and do not round unless the problem explicitly asks. "
    "If the problem has multiple [ANS] blanks, count every blank and answer all of them in order. "
    "Do not box intermediate values. Use \\boxed{} exactly once, on the final line only. "
    "The final line must contain only the boxed answer, with multiple answers separated by commas."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Use the <think> section for a short justification, then output only the chosen letter inside a single \\boxed{}. "
    "The boxed answer must be the last line and must contain no extra words. "
    "Before finalizing, verify calculations, option matching, and formatting mistakes. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

FEWSHOT_MATH = """Here are solved examples of the required answer style.

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
\boxed{\frac{3}{4}}

Now solve the next problem in the same format.
"""

FEWSHOT_MCQ = """Here are solved examples of the required answer style.

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


def build_few_shot_block(options: Optional[list]) -> str:
    return FEWSHOT_MCQ if options else FEWSHOT_MATH


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

        Example 3
        Problem: Give the exact fraction remaining after 36 years if the half-life is 31 years. [ANS]
        <think>
        This is exponential decay. The exact fraction remaining is (1/2)^(36/31). Since exact form is requested, do not convert to decimal.
        </think>
        \boxed{(1/2)^{36/31}}

        Example 4
        Problem: Fill the table values: x values [ANS], x^2 values [ANS], sum [ANS], average [ANS], square root [ANS]
        <think>
        There are five blanks, so every requested table entry must be included in order.
        </think>
        \boxed{-10, 100, -9, 81, 304, 60.8, 7.797}

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



def build_few_shot_block(options: Optional[list]) -> str:
    return FEWSHOT_MCQ if options else FEWSHOT_MATH


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_prompt = f"{build_few_shot_block(options)}\n\nQuestion:\n{question}\n\nOptions:\n{opts_text}"
        return SYSTEM_PROMPT_MCQ, user_prompt
    user_prompt = f"{build_few_shot_block(options)}\n\nQuestion:\n{question}"
    return SYSTEM_PROMPT_MATH, user_prompt


# Verify with samples
for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
    sys_p, usr_p = build_prompt(item["question"], item.get("options"))
    print(f"── {label} user prompt (first 200 chars) ──")
    print(usr_p[:200], "...\n")

#load model

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

MAX_TOKENS = 16384   # doubled generation budget

llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    gpu_memory_utilization=0.9,   # lower this
    max_model_len=24576,            # allow longer prompts + doubled generation
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


# -------- GENERATE + SCORE + SAVE (CRASH SAFE) --------

SAVE_EVAL = False
BATCH_SIZE = 5

out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

count_path = out_path.with_suffix(".count.txt")
# Resume point
if count_path.exists():
    start_idx = int(count_path.read_text().strip())
else:
    start_idx = 0

run_end_idx = len(data)
print(f"Starting full run at index {start_idx}; ending before {run_end_idx}", flush=True)

# -------- SCORING HELPERS --------

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

# -------- MAIN LOOP --------

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

    system, user = build_prompt(item["question"], item.get("options"))
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    output = llm.generate([prompt_text], sampling_params=sampling_params)
    response = output[0].outputs[0].text.strip()
    print(f"[{item_num}/{total_items}] Response preview: {response[:120].replace(chr(10), ' ')}", flush=True)

    # -------- SCORING --------

    is_mcq = bool(item.get("options"))
    # gold = item["answer"]
    gold = 1

    r = {
        "id": item.get("id"),
        "is_mcq": is_mcq,
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

    # -------- SAVE EVERY BATCH --------

    if len(pending_results) == BATCH_SIZE or idx == run_end_idx - 1:
        with open(out_path, "a") as f:
            for r in pending_results:
                if SAVE_EVAL:
                    record = {
                        "id": r["id"],
                        "is_mcq": r["is_mcq"],
                        "gold": r["gold"],
                        "response": r["response"],
                        "correct": r["correct"],
                    }
                else:
                    record = {
                        "id": r["id"],
                        "is_mcq": r["is_mcq"],
                        "response": r["response"],
                    }

                f.write(json.dumps(record) + "\n")

        saved_count = idx + 1
        count_path.write_text(str(saved_count))

        print(f"Saved through item {saved_count}; batch size {len(pending_results)}", flush=True)
        pending_results = []



#score

results = [json.loads(line) for line in open(OUTPUT_PATH)]
print(f"Scoring complete. {len(results)} results.", flush=True)

mcq_res  = [r for r in results if r["is_mcq"]]
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
