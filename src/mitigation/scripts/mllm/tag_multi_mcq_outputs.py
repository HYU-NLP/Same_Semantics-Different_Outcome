#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


ORDER = ["Both", "Internal", "Text", "Image", "Mixed", "None"]
STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "is", "are", "was", "were",
    "county", "city", "state", "province", "prefecture", "district",
}


def normalize(text):
    text = "" if text is None else str(text)
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def acronym(text):
    words = [
        w for w in normalize(text).split()
        if w not in {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to"}
    ]
    if len(words) < 2:
        return ""
    return "".join(w[0] for w in words if w)


def answer_spans(text):
    raw = "" if text is None else str(text).strip()
    if not raw:
        return []
    spans = []
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
                spans.append(m.group(1).strip(" .,:;\"'"))
        if ":" in chunk:
            spans.append(chunk.split(":", 1)[1].strip(" .,:;\"'"))
    return [s for s in spans if s]


def candidate_variants(values, derive_spans=False):
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
        if derive_spans:
            for span in answer_spans(raw):
                variants.add(span)
                acro = acronym(span)
                if len(acro) >= 2:
                    variants.add(acro)
        norm = normalize(raw)
        if norm:
            variants.add(norm)
        acro = acronym(raw)
        if len(acro) >= 2:
            variants.add(acro)
        for inside in re.findall(r"\(([^)]+)\)", raw):
            variants.add(inside)
    return sorted({normalize(v) for v in variants if normalize(v)}, key=len, reverse=True)


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


def single_tag(answer, internal_cands, text_cands, image_cands):
    hits = []
    if mentions(answer, internal_cands):
        hits.append("Internal")
    if mentions(answer, text_cands):
        hits.append("Text")
    if mentions(answer, image_cands):
        hits.append("Image")
    if len(hits) >= 2:
        return "Both"
    if len(hits) == 1:
        return hits[0]
    return "None"


def final_tag(tags):
    if not tags:
        return "None"
    first = tags[0]
    return first if all(tag == first for tag in tags) else "Mixed"


def pct(count, total):
    return round((count / total * 100) if total else 0.0, 2)


def candidates_for(row, qnum):
    internal = candidate_variants([row.get("memory_answer"), row.get("ground_truth"), row.get("memory_short_answer")], True)
    c1 = candidate_variants([row.get("counter_answer")], True)
    c2 = candidate_variants([row.get("counter_answer2")], True)
    if qnum == 1:
        return internal, c1, c2  # counter_answer=Text, counter_answer2=Image
    return internal, c2, c1      # counter_answer2=Text, counter_answer=Image


def tag_field(row, field, qnum):
    internal, text, image = candidates_for(row, qnum)
    answer_tags = [single_tag(ans, internal, text, image) for ans in (row.get(field) or [])]
    return answer_tags, final_tag(answer_tags), {"internal": internal, "text": text, "image": image}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--field-prefix", default="output")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = json.load(in_path.open(encoding="utf-8"))

    prefix = args.field_prefix.strip().rstrip("-")
    itq_group = f"{prefix}-itq" if prefix else "itq"
    tiq_group = f"{prefix}-tiq" if prefix else "tiq"
    counts = {itq_group: Counter(), tiq_group: Counter()}
    tagged_rows = []
    for row in rows:
        tagged = dict(row)
        for field, group, qnum in [
            (f"{prefix}-itq1" if prefix else "itq1", itq_group, 1),
            (f"{prefix}-itq2" if prefix else "itq2", itq_group, 2),
            (f"{prefix}-tiq1" if prefix else "tiq1", tiq_group, 1),
            (f"{prefix}-tiq2" if prefix else "tiq2", tiq_group, 2),
        ]:
            answer_tags, tag, cands = tag_field(row, field, qnum)
            tagged[f"{field}-tags"] = answer_tags
            tagged[f"{field}-final_tag"] = tag
            tagged[f"{field}-match_candidates"] = cands
            counts[group][tag] += 1
        tagged_rows.append(tagged)

    n = len(rows) * 2
    summary = {}
    for group in (itq_group, tiq_group):
        summary[group] = {
            "n": n,
            "counts": {tag: counts[group][tag] for tag in ORDER},
            "percentages": {tag: pct(counts[group][tag], n) for tag in ORDER},
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"input": str(in_path), "summary": summary, "rows": tagged_rows}, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
