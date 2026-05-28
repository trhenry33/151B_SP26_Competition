#!/usr/bin/env python3
"""
Post-process model outputs to extract final answers following the grader's expectations.

Usage:
  python extract_boxed.py --in raw_outputs.jsonl --out submission.jsonl [--public public_eval.jsonl]

Input JSONL should contain objects with at least: {"id":..., "is_mcq":..., "response":..., "gold":... (optional)}
This script writes a cleaned `response` where possible by extracting the final boxed answer or falling back to last LaTeX/number.
"""
import re
import json
import argparse
from typing import List, Optional


BOXED_RE = re.compile(r"\\boxed\{")
LAST_LATEX_RE = re.compile(r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\)|\\\])", re.DOTALL)
NUMBER_RE = re.compile(r"-?\d*\.?\d+")


def normalize_answer_simple(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # strip surrounding dollars and whitespace
    s = s.strip()
    s = s.strip("$")
    return s


def extract_all_boxed(search_text: str) -> List[str]:
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

    # keep only last contiguous group (allow only simple separators between boxes)
    last_group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = search_text[entries[j][1]:entries[j + 1][0]]
        if re.match(r'^[\s,\$\.\;\:\-\&\\]*$', gap):
            last_group.insert(0, entries[j])
        else:
            break

    return [e[2] for e in last_group]


def remove_boxed_full(text: str) -> Optional[str]:
    # find last \boxed{...} and return inner content raw (no normalize)
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
    # Only consider content after last </think>
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text

    all_boxed = extract_all_boxed(search_text)
    if len(all_boxed) > 1:
        return ", ".join(all_boxed)
    elif len(all_boxed) == 1:
        return all_boxed[0]

    # fallback: last boxed anywhere
    content = remove_boxed_full(text)
    if content is not None:
        return normalize_answer_simple(content)

    # fallback: last LaTeX formula after reasoning
    matches = LAST_LATEX_RE.findall(search_text)
    if matches:
        return normalize_answer_simple(matches[-1])

    # fallback: last number
    matches = NUMBER_RE.findall(search_text.replace(",", ""))
    if matches:
        return matches[-1]

    # give up: return full response trimmed
    return normalize_answer_simple(search_text)


def process_jsonl(infile: str, outfile: str, public_out: Optional[str] = None):
    out_lines = []
    pub_lines = []
    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = obj.get('id')
            is_mcq = obj.get('is_mcq', False)
            resp = obj.get('response', '')
            gold = obj.get('gold')

            cleaned = extract_boxed_answer(resp)

            out_obj = {'id': rid, 'is_mcq': is_mcq, 'response': cleaned}
            out_lines.append(out_obj)

            if public_out is not None:
                correct = None
                if gold is not None:
                    # basic normalization compare
                    gnorm = normalize_answer_simple(gold if isinstance(gold, str) else (gold[0] if isinstance(gold, list) else str(gold)))
                    pnorm = normalize_answer_simple(cleaned)
                    correct = gnorm == pnorm
                pub_obj = {'id': rid, 'is_mcq': is_mcq, 'gold': gold, 'response': cleaned, 'correct': correct}
                pub_lines.append(pub_obj)

    with open(outfile, 'w', encoding='utf-8') as fo:
        for o in out_lines:
            fo.write(json.dumps(o, ensure_ascii=False) + '\n')

    if public_out is not None:
        with open(public_out, 'w', encoding='utf-8') as fo:
            for o in pub_lines:
                fo.write(json.dumps(o, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', dest='outfile', required=True)
    parser.add_argument('--public', dest='public', required=False)
    args = parser.parse_args()
    process_jsonl(args.infile, args.outfile, args.public)


if __name__ == '__main__':
    main()
