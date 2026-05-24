import json

INPUT = "../data/public.jsonl"
OUTPUT = "data/sft_train.jsonl"

def make_target(item):
    ans = item["answer"]
    if isinstance(ans, list):
        ans = ", ".join(map(str, ans))
    return f"<think>\nWe solve carefully and verify the result.\n</think>\n\\boxed{{{ans}}}"

with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        item = json.loads(line)

        if item.get("options"):
            labels = [chr(65 + i) for i in range(len(item["options"]))]
            opts = "\n".join(f"{l}. {o}" for l, o in zip(labels, item["options"]))
            prompt = f"Question:\n{item['question']}\n\nOptions:\n{opts}"
        else:
            prompt = f"Question:\n{item['question']}"

        row = {
            "messages": [
                {"role": "system", "content": "You are an expert mathematician. Solve carefully and put the final answer inside \\boxed{}."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": make_target(item)},
            ]
        }
        fout.write(json.dumps(row) + "\n")

print("saved", OUTPUT)