#!/usr/bin/env python3
"""Build preference pairs for misleading-image factual QA.

chosen: factual/original answer
rejected: visually suggested counter/substitute answer
"""

import argparse
import json
from pathlib import Path


def first_answer(value):
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    skipped = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            chosen = str(row.get("answer", "")).strip()
            rejected = first_answer(row.get("sub_answer", row.get("counter_answer", "")))
            if not chosen or not rejected or chosen.lower() == rejected.lower():
                skipped += 1
                continue
            out = {
                "question": row["question"],
                "image": row["image"],
                "chosen": chosen,
                "rejected": rejected,
                "answer": chosen,
                "original_idx": row.get("original_idx"),
                "source": "nq_swap_misleading_image_dpo",
                "org_answer": row.get("org_answer"),
                "sub_answer": row.get("sub_answer"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    summary = {"input": str(in_path), "output": str(out_path), "n": n, "skipped": skipped}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
