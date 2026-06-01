import csv, re

INPUT_CSV = "fewshot_submission.csv"
OUTPUT_CSV = "fewshot_submission_final2.csv"

def extract_boxed_all(text):
    out = []
    i = 0
    while True:
        start = text.find(r"\boxed{", i)
        if start == -1:
            break
        j = start + len(r"\boxed{")
        depth = 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            out.append(text[start + len(r"\boxed{"):j-1].strip())
        i = j
    return out

def clean(ans):
    ans = ans.strip()
    ans = ans.strip("$")
    ans = ans.replace("\\displaystyle", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    return ans.strip()

def repair_answer(text):
    after = text.split("</think>")[-1] if "</think>" in text else text
    boxes_after = extract_boxed_all(after)
    boxes_all = extract_boxed_all(text)

    # Best case: use final boxed answer after thinking.
    if boxes_after:
        return clean(boxes_after[-1])

    # Otherwise use final boxed answer anywhere.
    if boxes_all:
        return clean(boxes_all[-1])

    # Final Answer line fallback.
    m = re.search(r"Final Answer\s*:?\s*(.*)", after, re.I | re.S)
    if m:
        tail = m.group(1).strip()
        b = extract_boxed_all(tail)
        if b:
            return clean(b[-1])
        line = tail.splitlines()[0].strip()
        if line:
            return clean(line)

    # MCQ fallback: last standalone A-J near end.
    letters = re.findall(r"\b([A-J])\b", after)
    if letters:
        return letters[-1]

    # Multi-number fallback: collect numbers from last 1500 chars.
    tail = after[-1500:]
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", tail, flags=re.I)
    if nums:
        # If many numbers, keep last up to 5. This helps multi-blank answers.
        return ", ".join(nums[-5:])

    return None

with open(INPUT_CSV, newline="", encoding="utf-8") as fin, \
     open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=["id", "response"])
    writer.writeheader()

    for row in reader:
        ans = repair_answer(row["response"])

        if ans:
            # Strong repair: replace final response with compact answer.
            # Use this if leaderboard only extracts final answer.
            response = f"\\boxed{{{ans}}}"
        else:
            response = row["response"]

        writer.writerow({
            "id": row["id"],
            "response": response,
        })

print(f"saved {OUTPUT_CSV}")