import json

PUBLIC = "data/public.jsonl"
RESULTS = "results/best_fewshot_public.jsonl"
OUTPUT = "data/sft_train_good_outputs.jsonl"

questions = {}
for line in open(PUBLIC):
    item = json.loads(line)
    questions[item["id"]] = item

with open(RESULTS) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        r = json.loads(line)

        if not r.get("correct"):
            continue

        item = questions[r["id"]]

        if item.get("options"):
            labels = [chr(65 + i) for i in range(len(item["options"]))]
            opts = "\n".join(f"{l}. {o}" for l, o in zip(labels, item["options"]))
            prompt = f"Question:\n{item['question']}\n\nOptions:\n{opts}"
        else:
            prompt = f"Question:\n{item['question']}"

        row = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert mathematician. Solve carefully and put the final answer inside \\boxed{}."
                },
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": r["response"]
                }
            ]
        }

        fout.write(json.dumps(row) + "\n")

print("saved", OUTPUT)