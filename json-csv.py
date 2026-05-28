import json
import csv

INPUT_JSONL = "lora_private.jsonl"   # change if needed
OUTPUT_CSV = "submission_lora.csv"

rows = []

with open(INPUT_JSONL, "r") as f:
    for line in f:
        r = json.loads(line)
        rows.append({
            "id": r["id"],
            "response": r["response"],
        })

rows.sort(key=lambda x: x["id"])

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "response"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")

import pandas as pd

df = pd.read_csv("submission_lora.csv")
print(df.head())
print(len(df))