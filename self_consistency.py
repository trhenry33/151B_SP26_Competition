import numpy
print(numpy.__version__)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

print("imports work")

import json
import os
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID = "0"
DATA_PATH = "data/public.jsonl"
OUTPUT_PATH = "results/self_consistency_public_20.jsonl"
K = 8

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


data = [json.loads(line) for line in open(DATA_PATH, encoding="utf-8")]

n_mcq = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options") for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

mcq_sample = next(d for d in data if d.get("options"))
free_sample = next(d for d in data if not d.get("options"))

print("\n── MCQ sample ──")
print(json.dumps(mcq_sample, indent=2))
print("\n── Free-form sample ──")
print(json.dumps(free_sample, indent=2))


SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Before finalizing, verify calculations, algebra, and formatting mistakes. Correct any errors before giving the final boxed answer. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
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
\\boxed{\\frac{3}{4}}

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


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_prompt = f"{build_few_shot_block(options)}\n\nQuestion:\n{question}\n\nOptions:\n{opts_text}"
        return SYSTEM_PROMPT_MCQ, user_prompt
    user_prompt = f"{build_few_shot_block(options)}\n\nQuestion:\n{question}"
    return SYSTEM_PROMPT_MATH, user_prompt


for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
    sys_p, usr_p = build_prompt(item["question"], item.get("options"))
    print(f"── {label} user prompt (first 200 chars) ──")
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
    n=K,
    max_tokens=MAX_TOKENS,
    temperature=0.8,
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
print(f"Starting self-consistency public run at index {start_idx}; ending before {run_end_idx}", flush=True)


sys.path.insert(0, ".")
from judger import Judger

judger = Judger(strict_extract=False)


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


BOOL_MAP = {
    "true": "True",
    "false": "False",
    "yes": "True",
    "no": "False",
    "y": "True",
    "n": "False",
    "t": "True",
    "f": "False",
}


def strip_wrappers(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("$", "")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("\n", " ")
    return text.strip()


def simplify_latex_fractions(text: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r"\1/\2", text)
    return text


def numeric_to_canonical(text: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None

    candidate = simplify_latex_fractions(candidate)
    candidate = candidate.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    candidate = candidate.replace(" ", "")

    if re.fullmatch(r"[+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?", candidate):
        try:
            value = Fraction(Decimal(candidate))
            value = value.limit_denominator(10**6)
            return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        except (InvalidOperation, ZeroDivisionError):
            return candidate

    if re.fullmatch(r"[+-]?\d+/[+-]?\d+", candidate):
        try:
            numerator_text, denominator_text = candidate.split("/", 1)
            value = Fraction(int(numerator_text), int(denominator_text))
            return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        except (ValueError, ZeroDivisionError):
            return candidate

    return None


def normalize_atom(text: str) -> str | None:
    s = strip_wrappers(judger.normalize_answer(text))
    if not s:
        return None

    s = s.strip(".,;:!")
    s_lower = s.lower()

    if s_lower in BOOL_MAP:
        return BOOL_MAP[s_lower]

    if re.fullmatch(r"[a-j]", s_lower):
        return s.upper()

    numeric = numeric_to_canonical(s)
    if numeric is not None:
        return numeric

    s = s.replace("\u2212", "-")
    s = s.replace("\t", "")
    s = re.sub(r"\s+", "", s)
    return s if s else None


def normalize_candidate(text: str) -> str | None:
    candidate = strip_wrappers(text)
    if not candidate:
        return None

    candidate = judger.extract_explicit_ans(candidate) or candidate
    candidate = strip_wrappers(candidate)
    if not candidate:
        return None

    parts = judger.split_by_comma(candidate) if "," in candidate else [candidate]
    normalized_parts = []
    for part in parts:
        normalized = normalize_atom(part)
        if normalized is None:
            return None
        normalized_parts.append(normalized)

    normalized_text = ", ".join(normalized_parts)
    if not normalized_text:
        return None

    # Discard obvious malformed outputs.
    alpha_words = re.findall(r"[A-Za-z]+", normalized_text)
    if len(normalized_text) > 120:
        return None
    if len(alpha_words) > 6 and not any(ch in normalized_text for ch in ["=", "\\", "/", "^", "_", "(", ")", "[", "]", "{"]):
        return None

    return normalized_text


def vote_majority(votes: list[str]) -> tuple[str, Counter]:
    counts = Counter(votes)
    if not counts:
        return "", counts

    first_seen = {}
    for idx, vote in enumerate(votes):
        first_seen.setdefault(vote, idx)

    winner = max(counts.keys(), key=lambda vote: (counts[vote], -first_seen[vote]))
    return winner, counts


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

    output = llm.generate([prompt_text], sampling_params=sampling_params)[0]
    raw_responses = [sample.text.strip() for sample in output.outputs]
    normalized_votes = []
    rejected = []

    for sample_text in raw_responses:
        normalized = normalize_candidate(sample_text)
        if normalized is None:
            rejected.append(sample_text)
            continue
        normalized_votes.append(normalized)

    final_response, vote_counts = vote_majority(normalized_votes)
    if not final_response:
        final_response = normalize_candidate(raw_responses[0]) or raw_responses[0].strip()

    print(
        f"[{item_num}/{total_items}] Votes: {dict(vote_counts)} -> {final_response}",
        flush=True,
    )

    is_mcq = bool(item.get("options"))
    gold = item.get("answer")

    r = {
        "id": item.get("id"),
        "is_mcq": is_mcq,
        "response": final_response,
        "raw_responses": raw_responses,
        "normalized_votes": normalized_votes,
        "vote_counts": dict(vote_counts),
        "rejected_responses": rejected,
    }

    if SAVE_EVAL:
        if is_mcq:
            correct = score_mcq(final_response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=final_response,
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
                        "response": r["response"],
                        "raw_responses": r["raw_responses"],
                        "normalized_votes": r["normalized_votes"],
                        "vote_counts": r["vote_counts"],
                        "rejected_responses": r["rejected_responses"],
                        "correct": r["correct"],
                    }
                else:
                    record = {
                        "id": r["id"],
                        "is_mcq": r["is_mcq"],
                        "response": r["response"],
                        "raw_responses": r["raw_responses"],
                        "normalized_votes": r["normalized_votes"],
                        "vote_counts": r["vote_counts"],
                        "rejected_responses": r["rejected_responses"],
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
