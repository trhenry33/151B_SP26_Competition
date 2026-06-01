import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
INPUT_PATH = "data/public.jsonl"
OUTPUT_PATH = "starter_results_5.jsonl"
N_TEST = 5

SYSTEM_PROMPT = """You are solving math problems for an autograder.

You may reason briefly inside <think>...</think>.

CRITICAL FORMAT RULE:
After the final </think>, output exactly one line:
\\boxed{answer}

Do not write explanation, prose, markdown, equations, or text after </think>.
Do not write anything after the boxed answer.

For multiple choice, answer with only the letter, like \\boxed{A}.
For free-form, answer with a parseable expression, like \\boxed{42} or \\boxed{\\frac{5}{8}}.
"""

def get_question_text(q):
    for key in ["question", "problem", "prompt"]:
        if key in q:
            return q[key]
    return str(q)

def get_choices(q):
    for key in ["choices", "options", "answer_choices"]:
        if key in q:
            return q[key]
    return None

def build_prompt(q):
    question = get_question_text(q)
    choices = get_choices(q)
    is_mcq = q.get("is_mcq", choices is not None)

    if is_mcq and choices:
        if isinstance(choices, dict):
            choices_text = "\n".join([f"{k}. {v}" for k, v in choices.items()])
        elif isinstance(choices, list):
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            choices_text = "\n".join([f"{letters[i]}. {choice}" for i, choice in enumerate(choices)])
        else:
            choices_text = str(choices)

        return f"""{SYSTEM_PROMPT}

Question:
{question}

Choices:
{choices_text}

Solve the problem. Remember: after </think>, output only \\boxed{{letter}}.
"""
    else:
        return f"""{SYSTEM_PROMPT}

Question:
{question}

Solve the problem. Remember: after </think>, output only \\boxed{{answer}}.
"""

def clean_response(text):
    # Keep only generated text after the last assistant prompt if possible
    text = text.strip()

    # If model produced a boxed answer, force the final response to end cleanly with only that box after </think>.
    boxes = re.findall(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if boxes:
        final_box = boxes[-1]
        if "</think>" in text:
            reasoning = text.split("</think>")[-2] if text.count("</think>") >= 1 else ""
            before = text.rsplit("</think>", 1)[0]
            return before.strip() + "\n</think>\n" + final_box
        return "</think>\n" + final_box

    return text

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading 8-bit model...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)

rows = []
with open(INPUT_PATH, "r") as f:
    for i, line in enumerate(f):
        if i >= N_TEST:
            break
        rows.append(json.loads(line))

with open(OUTPUT_PATH, "w") as out_f:
    for q in tqdm(rows):
        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if decoded.startswith(prompt):
            response = decoded[len(prompt):].strip()
        else:
            response = decoded.strip()

        response = clean_response(response)

        result = {
            "id": q.get("id"),
            "is_mcq": q.get("is_mcq", get_choices(q) is not None),
            "response": response,
        }

        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

print(f"Wrote {OUTPUT_PATH}")
