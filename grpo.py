import numpy
print(numpy.__version__)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

print("imports work")


#imports
#baseline is 60% with first 10
#basline is 50% with first 20 - MC: 4/9 FR: 6/11
#reflection is same as baseline

#to test: rm results/count.txt and change the output path on this file to start a new file to check.

#travis test path so far: baseline -> reflective

import json, os, re, sys
from collections import Counter
from pathlib import Path
from typing import Optional

MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results/grpo_style_20_items_updated.jsonl"


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

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem carefully and briefly. "
    "Use the <think> section for working, then give the final answer only inside a single \\boxed{}. "
    "The boxed answer must be the last line and must contain no extra words. "
    "Before finalizing, verify calculations, algebra, and formatting mistakes. "
    "If the problem has multiple sub-answers, separate them by commas inside one \\boxed{}, "
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

MAX_TOKENS = 4096

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

# GRPO-style grouped sampling: generate multiple completions per question.
MCQ_SAMPLES = 3
FREE_INITIAL_SAMPLES = 1
FREE_FALLBACK_SAMPLES = 2
PRIMARY_TEMPERATURE = 0.7
FALLBACK_TEMPERATURE = 0.9
VERIFIER_TEMPERATURE = 0.0
VERIFIER_MAX_TOKENS = 256


def sample_responses(prompt_text: str, num_samples: int, temperature: float, max_tokens: int = MAX_TOKENS) -> list[str]:
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        n=num_samples,
    )
    output = llm.generate([prompt_text], sampling_params=sampling_params)
    return [candidate.text.strip() for candidate in output[0].outputs]


# -------- GENERATE + SCORE + SAVE (CRASH SAFE) --------

SAVE_EVAL = True
BATCH_SIZE = 5
RUN_LIMIT = 20

out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)

count_path = out_path.parent / "count.txt"

# Resume point
if count_path.exists():
    start_idx = int(count_path.read_text().strip())
else:
    start_idx = 0

run_end_idx = min(len(data), start_idx + RUN_LIMIT)
print(f"Starting run at index {start_idx}; ending before {run_end_idx} (limit {RUN_LIMIT} items)", flush=True)
print(
    f"Using adaptive grouped sampling (MCQ={MCQ_SAMPLES}, free-form initial={FREE_INITIAL_SAMPLES}, fallback={FREE_FALLBACK_SAMPLES}).",
    flush=True,
)

# -------- SCORING HELPERS --------

def normalize_answer_simple(s: str) -> str:
    if s is None:
        return ""
    return s.strip().strip("$")


BOXED_RE = re.compile(r"\\boxed\{")
LAST_LATEX_RE = re.compile(r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\)|\\\])", re.DOTALL)
NUMBER_RE = re.compile(r"-?\d*\.?\d+")


def extract_all_boxed(search_text: str) -> list[str]:
    entries = []
    start = 0
    while True:
        m = BOXED_RE.search(search_text, start)
        if not m:
            break
        idx = m.start()
        brace_start = m.end()
        depth = 1
        i = brace_start
        while i < len(search_text) and depth > 0:
            if search_text[i] == '{':
                depth += 1
            elif search_text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            content = search_text[brace_start:i - 1]
            if content:
                entries.append((idx, i, normalize_answer_simple(content)))
        start = i

    if not entries:
        return []

    last_group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = search_text[entries[j][1]:entries[j + 1][0]]
        if re.match(r'^[\s,\$\.\;\:\-\&\\]*$', gap):
            last_group.insert(0, entries[j])
        else:
            break

    return [e[2] for e in last_group]


def remove_boxed_full(text: str) -> Optional[str]:
    m = None
    for mm in re.finditer(r"\\boxed\{", text):
        m = mm
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1]
    return None


def extract_boxed_answer(text: str) -> str:
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    all_boxed = extract_all_boxed(search_text)
    if len(all_boxed) > 1:
        return ", ".join(all_boxed)
    elif len(all_boxed) == 1:
        return all_boxed[0]

    content = remove_boxed_full(text)
    if content is not None:
        return normalize_answer_simple(content)

    matches = LAST_LATEX_RE.findall(search_text)
    if matches:
        return normalize_answer_simple(matches[-1])

    matches = NUMBER_RE.findall(search_text.replace(",", ""))
    if matches:
        return matches[-1]

    return normalize_answer_simple(search_text)


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_answer(response: str, gold) -> bool:
    if isinstance(gold, list):
        gold_list = [normalize_answer_simple(x if isinstance(x, str) else str(x)) for x in gold]
        pred = normalize_answer_simple(extract_boxed_answer(response))
        return pred in gold_list

    gold_str = str(gold).strip().upper()
    if len(gold_str) == 1 and gold_str.isalpha():
        return extract_letter(response) == gold_str
    return normalize_answer_simple(extract_boxed_answer(response)) == normalize_answer_simple(str(gold))


def reward_from_response(response: str, gold) -> float:
    """Simple GRPO-style reward: correct answer earns 1.0, wrong answer earns 0.0, malformed output gets a small penalty."""
    extracted = extract_letter(response) if (isinstance(gold, str) and len(str(gold).strip()) == 1 and str(gold).strip().isalpha()) else extract_boxed_answer(response)
    if not extracted:
        return -0.25
    return 1.0 if score_answer(response, gold) else 0.0


def compute_group_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
    std_reward = variance ** 0.5
    if std_reward == 0:
        return [0.0 for _ in rewards]
    return [(r - mean_reward) / std_reward for r in rewards]


def pick_best_sample(responses: list[str], rewards: list[float]) -> tuple[int, str]:
    best_idx = 0
    best_reward = rewards[0]
    for idx, reward in enumerate(rewards):
        if reward > best_reward:
            best_idx = idx
            best_reward = reward
    return best_idx, responses[best_idx]


def format_confidence(response: str, extracted_answer: str, is_mcq: bool) -> float:
    confidence = 0.0
    if extracted_answer:
        confidence += 1.0
    if is_mcq:
        if re.search(r"\\boxed\{[A-Z]\}", response):
            confidence += 0.5
        elif extract_letter(response):
            confidence += 0.15
    else:
        if "\\boxed" in response:
            confidence += 0.5
        if response.count("\n") <= 4:
            confidence += 0.1
        if len(response) > 400:
            confidence -= 0.1
    return confidence


def candidate_reward(response: str, extracted_answer: str, consensus_count: int, is_mcq: bool) -> float:
    reward = format_confidence(response, extracted_answer, is_mcq)
    if consensus_count > 1:
        reward += 0.25 * (consensus_count - 1)
    return reward


def should_add_fallback(responses: list[str], extracted_answers: list[str], is_mcq: bool) -> bool:
    if is_mcq:
        if any(not answer for answer in extracted_answers):
            return True
        return len(set(extracted_answers)) > 1 and Counter(extracted_answers).most_common(1)[0][1] == 1

    if not extracted_answers or not extracted_answers[0]:
        return True
    first_response = responses[0] if responses else ""
    return "\\boxed" not in first_response or len(first_response.splitlines()) > 8


def build_verifier_prompt(question: str, options: Optional[list], candidate_responses: list[str], candidate_answers: list[str], is_mcq: bool) -> str:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        question_block = f"Question:\n{question}\n\nOptions:\n{opts_text}"
    else:
        question_block = f"Question:\n{question}"

    candidate_blocks = []
    for idx, (response, answer) in enumerate(zip(candidate_responses, candidate_answers), start=1):
        candidate_blocks.append(
            f"Candidate {idx}:\nResponse:\n{response}\nExtracted answer: {answer or '[none]'}"
        )

    verifier_intro = (
        "You are a strict math verifier. Compare the candidates and select the single best final answer. "
        "If the candidates disagree, prefer the one that is mathematically correct and in the right output format. "
        "Return only the final answer in the required format, with no explanation."
    )
    if is_mcq:
        verifier_intro += " For multiple-choice problems, output only the letter inside \\boxed{}."
    else:
        verifier_intro += " For free-form problems, output only the final answer inside \\boxed{}."

    return f"{verifier_intro}\n\n{question_block}\n\n" + "\n\n".join(candidate_blocks)


def verifier_choose(question: str, options: Optional[list], candidate_responses: list[str], candidate_answers: list[str], is_mcq: bool) -> str:
    verifier_prompt = build_verifier_prompt(question, options, candidate_responses, candidate_answers, is_mcq)
    verifier_chat_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a strict math verifier."},
            {"role": "user", "content": verifier_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    verifier_response = sample_responses(
        verifier_chat_prompt,
        num_samples=1,
        temperature=VERIFIER_TEMPERATURE,
        max_tokens=VERIFIER_MAX_TOKENS,
    )[0]
    return verifier_response


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

    is_mcq = bool(item.get("options"))
    gold = item["answer"]

    if is_mcq:
        print(f"[{item_num}/{total_items}] Generating {MCQ_SAMPLES} MCQ candidates...", flush=True)
        responses = sample_responses(prompt_text, num_samples=MCQ_SAMPLES, temperature=PRIMARY_TEMPERATURE)
    else:
        print(f"[{item_num}/{total_items}] Generating 1 free-form candidate...", flush=True)
        responses = sample_responses(prompt_text, num_samples=FREE_INITIAL_SAMPLES, temperature=PRIMARY_TEMPERATURE)

    extracted_samples = [extract_letter(response) if is_mcq else extract_boxed_answer(response) for response in responses]
    needs_fallback = should_add_fallback(responses, extracted_samples, is_mcq)

    if needs_fallback and not is_mcq:
        print(f"[{item_num}/{total_items}] Free-form answer looks uncertain; sampling {FREE_FALLBACK_SAMPLES} more candidates...", flush=True)
        extra_responses = sample_responses(prompt_text, num_samples=FREE_FALLBACK_SAMPLES, temperature=FALLBACK_TEMPERATURE)
        responses.extend(extra_responses)
        extracted_samples = [extract_letter(response) if is_mcq else extract_boxed_answer(response) for response in responses]
    elif needs_fallback and is_mcq:
        print(f"[{item_num}/{total_items}] MCQ candidates disagree; keeping verifier path active.", flush=True)

    print(f"[{item_num}/{total_items}] Generation finished with {len(responses)} candidates.", flush=True)

    answer_counts = Counter(extracted_samples)
    rewards = [candidate_reward(response, extracted, answer_counts[extracted], is_mcq) for response, extracted in zip(responses, extracted_samples)]
    advantages = compute_group_advantages(rewards)
    best_idx, best_response = pick_best_sample(responses, rewards)

    reward_summary = ", ".join(f"{i}:{reward:.2f}" for i, reward in enumerate(rewards))
    advantage_summary = ", ".join(f"{i}:{adv:.2f}" for i, adv in enumerate(advantages))

    print(f"[{item_num}/{total_items}] Rewards: {reward_summary}", flush=True)
    print(f"[{item_num}/{total_items}] Advantages: {advantage_summary}", flush=True)
    print(f"[{item_num}/{total_items}] Best sample index: {best_idx}", flush=True)

    if is_mcq:
        top_count = answer_counts.most_common(1)[0][1] if answer_counts else 0
        if top_count == 1 or needs_fallback:
            print(f"[{item_num}/{total_items}] Running verifier rerank for MCQ...", flush=True)
            verifier_response = verifier_choose(item["question"], item.get("options"), responses, extracted_samples, is_mcq=True)
            verifier_answer = extract_letter(verifier_response)
            if verifier_answer:
                best_response = verifier_response
                best_idx = -1
                print(f"[{item_num}/{total_items}] Verifier selected MCQ answer: {verifier_answer}", flush=True)
    else:
        print(f"[{item_num}/{total_items}] Running verifier rerank for free-form...", flush=True)
        verifier_response = verifier_choose(item["question"], item.get("options"), responses, extracted_samples, is_mcq=False)
        verifier_answer = extract_boxed_answer(verifier_response)
        if verifier_answer:
            best_response = verifier_response
            best_idx = -1
            print(f"[{item_num}/{total_items}] Verifier selected free-form answer: {verifier_answer}", flush=True)

    # -------- SCORING --------

    r = {
        "id": item.get("id"),
        "is_mcq": is_mcq,
        "response": best_response,
        "samples": responses,
        "sample_answers": extracted_samples,
        "rewards": rewards,
        "advantages": advantages,
        "best_sample_idx": best_idx,
    }

    if SAVE_EVAL:
        if is_mcq:
            correct = score_answer(best_response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=best_response,
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
                        "samples": r["samples"],
                        "sample_answers": r["sample_answers"],
                        "rewards": r["rewards"],
                        "advantages": r["advantages"],
                        "best_sample_idx": r["best_sample_idx"],
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