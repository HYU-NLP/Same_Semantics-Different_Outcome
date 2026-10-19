#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


ORDER = ["Both", "Internal", "External", "Mixed", "None"]


def normalize(text):
    text = "" if text is None else str(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def acronym(text):
    norm = normalize(text)
    words = [w for w in norm.split() if w not in {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to"}]
    if len(words) < 2:
        return ""
    return "".join(w[0] for w in words if w)



def counter_answer_spans(text):
    """Derive answer-like spans from a full counter_answer without using counter_short_answer."""
    raw = "" if text is None else str(text).strip()
    if not raw:
        return []
    spans = []
    # Keep the full sentence, plus predicate/object tails after common answer-introducing patterns.
    patterns = [
        r"\b(?:is|are|was|were|be|been|being|became|becomes)\b\s+(.+)$",
        r"\b(?:occupation|profession|job|role|genre|location|place|country|city|language|nationality|creator|author|writer|director|screenwriter|composer|producer|capital)\b\s+(?:is|are|was|were)\s+(.+)$",
        r"\b(?:located|born|founded|created|written|directed|produced|composed|published|released)\s+(?:in|by|at|on)\s+(.+)$",
    ]
    for chunk in re.split(r"[.;!?]\s*", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        for pattern in patterns:
            m = re.search(pattern, chunk, flags=re.IGNORECASE)
            if m:
                tail = m.group(1).strip(" .,:;\"'")
                if tail:
                    spans.append(tail)
        # Possessive templates such as "X's occupation is novelist" are covered above,
        # but this also catches colon-style answers.
        if ":" in chunk:
            tail = chunk.split(":", 1)[1].strip(" .,:;\"'")
            if tail:
                spans.append(tail)
    return spans

def candidate_variants(values, derive_counter_answer_spans=False):
    if isinstance(values, str):
        values = [values]
    variants = set()
    for value in values or []:
        if value is None:
            continue
        raw = str(value).strip()
        if not raw:
            continue
        variants.add(raw)
        if derive_counter_answer_spans:
            for span in counter_answer_spans(raw):
                variants.add(span)
                acro_span = acronym(span)
                if len(acro_span) >= 2:
                    variants.add(acro_span)
        norm = normalize(raw)
        if norm:
            variants.add(norm)
        acro = acronym(raw)
        if len(acro) >= 2:
            variants.add(acro)
        for inside in re.findall(r"\(([^)]+)\)", raw):
            inside = inside.strip()
            if inside:
                variants.add(inside)
                inside_norm = normalize(inside)
                if inside_norm:
                    variants.add(inside_norm)
    return sorted({normalize(v) for v in variants if normalize(v)}, key=len, reverse=True)


STOPWORDS = {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "is", "are", "was", "were", "county", "city", "state", "province", "prefecture", "district"}


def content_tokens(text):
    return [tok for tok in normalize(text).split() if tok not in STOPWORDS and not tok.isdigit()]


def token_subset_match(ans, cand):
    ans_tokens = set(content_tokens(ans))
    cand_tokens = set(content_tokens(cand))
    if len(ans_tokens) < 2 or len(cand_tokens) < 2:
        return False
    return ans_tokens.issubset(cand_tokens) or cand_tokens.issubset(ans_tokens)


def mentions(answer, candidates):
    ans = normalize(answer)
    if not ans:
        return False
    compact_ans = ans.replace(" ", "")
    for cand in candidates:
        if not cand:
            continue
        compact_cand = cand.replace(" ", "")
        if len(compact_cand) < 2:
            continue
        if cand in ans or ans in cand:
            return True
        if compact_cand in compact_ans or compact_ans in compact_cand:
            return True
        if token_subset_match(ans, cand):
            return True
    return False


def single_tag(answer, internal_candidates, external_candidates):
    internal = mentions(answer, internal_candidates)
    external = mentions(answer, external_candidates)
    if internal and external:
        return "Both"
    if internal:
        return "Internal"
    if external:
        return "External"
    return "None"


def final_tag(tags):
    if not tags:
        return "None"
    first = tags[0]
    if all(tag == first for tag in tags):
        return first
    return "Mixed"


def row_candidates(row, external_field):
    internal_values = []
    internal_values.extend(row.get("ground_truth") or [])

    external_values = []
    if row.get(external_field):
        external_values.append(row[external_field])

    return candidate_variants(internal_values), candidate_variants(
        external_values, derive_counter_answer_spans=(external_field == "counter_answer")
    )


def pct(count, total):
    return round((count / total * 100) if total else 0.0, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--external_field", default="counter_short_answer")
    parser.add_argument(
        "--fields",
        default="output-text,output-image",
        help="Comma-separated output fields to tag. Defaults preserve the old text/image evaluator behavior.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    with in_path.open(encoding="utf-8") as f:
        rows = json.load(f)

    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    if not fields:
        raise ValueError("--fields must contain at least one field name.")

    tagged_rows = []
    counts = {field: Counter() for field in fields}

    for row in rows:
        internal_candidates, external_candidates = row_candidates(row, args.external_field)
        tagged = dict(row)
        tagged["_match_candidates"] = {
            "internal": internal_candidates,
            "external": external_candidates,
            "external_field": args.external_field,
        }
        for field in fields:
            answers = row.get(field) or []
            answer_tags = [single_tag(ans, internal_candidates, external_candidates) for ans in answers]
            tag = final_tag(answer_tags)
            tagged[f"{field}-tags"] = answer_tags
            tagged[f"{field}-final_tag"] = tag
            counts[field][tag] += 1
        tagged_rows.append(tagged)

    summary = {}
    n = len(rows)
    for field in fields:
        summary[field] = {
            "n": n,
            "counts": {tag: counts[field][tag] for tag in ORDER},
            "percentages": {tag: pct(counts[field][tag], n) for tag in ORDER},
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"input": str(in_path), "summary": summary, "rows": tagged_rows}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
