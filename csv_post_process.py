import csv
import re
from pathlib import Path

INPUT_CSV = "submission_lora.csv"
OUTPUT_CSV = "lora_submission_final.csv"

def extract_last_boxed(text):
    matches = []
    start = 0

    while True:
        idx = text.find(r"\boxed{", start)
        if idx == -1:
            break

        i = idx + len(r"\boxed{")
        depth = 1

        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            matches.append(text[idx + len(r"\boxed{"): i - 1])

        start = i

    return matches[-1].strip() if matches else None


def guess_final_answer(text):
    # Prefer content after </think>
    if "</think>" in text:
        search = text.split("</think>")[-1]
    else:
        search = text

    boxed = extract_last_boxed(search)
    if boxed:
        return boxed

    boxed = extract_last_boxed(text)
    if boxed:
        return boxed

    # Try "Final Answer" style
    patterns = [
        r"Final Answer[:\s]*([^\n]+)",
        r"final answer is[:\s]*([^\n]+)",
        r"answer is[:\s]*([^\n]+)",
    ]

    for pat in patterns:
        m = re.search(pat, search, flags=re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            ans = ans.strip("$ .")
            return ans

    # MCQ fallback: last standalone capital letter
    letters = re.findall(r"\b([A-J])\b", search)
    if letters:
        return letters[-1]

    # Numeric fallback: last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", search.replace(",", ""))
    if nums:
        return nums[-1]

    return None


def postprocess_response(response):
    ans = guess_final_answer(response)

    if not ans:
        return response

    # Keep original response, but force a clean final boxed line
    return response.rstrip() + f"\n\n\\boxed{{{ans}}}"


with open(INPUT_CSV, newline="", encoding="utf-8") as fin, \
     open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=["id", "response"])
    writer.writeheader()

    for row in reader:
        row["response"] = postprocess_response(row["response"])
        writer.writerow({
            "id": row["id"],
            "response": row["response"],
        })

print(f"Saved postprocessed CSV to {OUTPUT_CSV}")