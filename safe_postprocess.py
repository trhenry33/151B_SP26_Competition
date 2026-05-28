#!/usr/bin/env python3
"""
Conservative postprocessor for competition submissions.

Goal:
- Preserve the raw response whenever extraction is uncertain.
- Only normalize when a clear final answer is present.

Usage:
  python safe_postprocess.py --in results/fewshot_examples_private_fullrun.jsonl --out boxed_fewshot_submission_fixed.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path


BOXED_RE = re.compile(r"\\boxed\{")
ANSWER_MARKERS = (
    "final answer",
    "answer is",
    "therefore",
    "thus",
    "so the answer",
    "so the final answer",
    "therefore,",
    "hence",
)
TRAILING_JUNK_RE = re.compile(r"[\s\.,;:!\?\)\]\}]+$")


def strip_wrappers(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("$", "")
    return text.strip()


def clean_candidate(text: str) -> str:
    text = strip_wrappers(text)
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac")
    text = text.replace("\\tfrac", "\\frac")
    text = text.replace("∶", ":")
    text = text.replace("，", ",")
    text = text.replace("\n", " ")
    text = TRAILING_JUNK_RE.sub("", text).strip()
    return text


def extract_boxed(text: str) -> str | None:
    """Extract the last well-formed boxed expression, or None if not found."""
    last = None
    for match in BOXED_RE.finditer(text):
        start = match.end()
        depth = 1
        idx = start
        while idx < len(text) and depth > 0:
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            last = text[start:idx - 1]
    return strip_wrappers(last) if last else None


def tail_after_marker(text: str) -> str | None:
    lower = text.lower()
    best_idx = -1
    for marker in ANSWER_MARKERS:
        idx = lower.rfind(marker)
        if idx > best_idx:
            best_idx = idx
    if best_idx == -1:
        return None
    tail = text[best_idx:]
    # keep only a short tail after the marker if it looks like a final answer
    tail = tail.split("\n")[-1].strip()
    if len(tail) > 120:
        return None
    return clean_candidate(tail)


def looks_like_answer_only(text: str) -> bool:
    t = strip_wrappers(text)
    if not t:
        return False
    if len(t) > 120:
        return False
    if re.fullmatch(r"[A-J]", t):
        return True
    if re.fullmatch(r"-?\d+(?:\.\d+)?", t):
        return True
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+", t):
        return True
    if re.fullmatch(r"\\?[A-Za-z0-9\\{}^_()/*+\-.,=\s]+", t) and len(t) <= 80:
        return True
    return False


def has_explicit_final_answer(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in ANSWER_MARKERS) or "\\boxed{" in text


def format_boxed_response(raw: str, candidate: str) -> str:
    candidate = clean_candidate(candidate)
    if not candidate:
        return raw
    return raw.rstrip() + "\n\nFinal answer: \\boxed{" + candidate + "}"


def normalize_response(resp: str) -> str:
    raw = strip_wrappers(resp)
    if not raw:
        return raw

    # If already answer-like, preserve it exactly.
    if looks_like_answer_only(raw):
        return raw

    # Prefer a clean boxed final answer if present.
    boxed = extract_boxed(raw)
    if boxed:
        boxed = clean_candidate(boxed)
        if boxed and (looks_like_answer_only(boxed) or len(boxed) <= 120):
            return format_boxed_response(raw, boxed)

    # If the model clearly states a final answer near the end, use the tail.
    tail = tail_after_marker(raw)
    if tail and (looks_like_answer_only(tail) or len(tail) <= 80):
        return format_boxed_response(raw, tail)

    # If the response is already short and looks answer-like after trimming,
    # keep it as-is; otherwise leave the raw reasoning untouched.
    if has_explicit_final_answer(raw):
        return raw

    # Otherwise preserve the original response to avoid damaging a good answer.
    return raw


def process_jsonl_to_csv(infile: Path, outfile: Path) -> int:
    rows = []
    with infile.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = obj.get("id")
            resp = normalize_response(obj.get("response", ""))
            rows.append({"id": rid, "response": resp})

    rows.sort(key=lambda r: int(r["id"]))
    with outfile.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    args = parser.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    count = process_jsonl_to_csv(infile, outfile)
    print(f"Saved {count} rows to {outfile}")


if __name__ == "__main__":
    main()
