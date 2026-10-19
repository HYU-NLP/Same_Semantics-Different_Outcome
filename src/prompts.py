import random

CHOICE_LETTERS = ["A", "B", "C", "D"]
NONE_OF_THEM = "None of them"

MCQ_TEMPLATE = '''Answer the following multiple-choice question based on your own knowledge.

Question: {query}

Choices:
{choices_str}

Instructions:
- Choose the option that best answers the question.
- Choose "{none_of_them}" only if none of the other choices is correct.
- Answer with only one letter: {letters}.
'''

INTERNAL_KNOWLEDGE_TEMPLATE = '''Answer the following question based on your own knowledge.

Question: {query}

Answer in 1–5 words. Do not use a full sentence.'''

EVIDENCE_TEMPLATE = '''{evidence}

Question: {query}

Answer in 1–5 words. Do not use a full sentence.'''


def first_ground_truth(sample):
    ground_truth = sample.get("ground_truth", "")
    if isinstance(ground_truth, list):
        return ground_truth[0] if ground_truth else ""
    return ground_truth

def normalize_letter(output):
    output = str(output).strip().upper()
    for letter in CHOICE_LETTERS:
        if output == letter or output.startswith(f"{letter}.") or output.startswith(f"{letter})"):
            return letter
    return output[:1] if output[:1] in CHOICE_LETTERS else ""

def build_options(sample):
    options = [
        {"tag": "ground_truth", "text": first_ground_truth(sample)},
        {"tag": "memory_short_answer", "text": sample.get("memory_short_answer", "")},
        {"tag": "counter_short_answer", "text": sample.get("counter_short_answer", "")},
        {"tag": "none_of_them", "text": NONE_OF_THEM},
    ]

    random.shuffle(options)
    return [
        {"letter": letter, "tag": option["tag"], "text": option["text"]}
        for letter, option in zip(CHOICE_LETTERS, options)
    ]

def mcq_prompt(query, choices):
    """Figure 8; identical formatting to ConflictQA/code/mcq/5_mcq.py."""
    return MCQ_TEMPLATE.format(
        query=query,
        choices_str="\n".join(f"{c['letter']}. {c['text']}" for c in choices),
        none_of_them=NONE_OF_THEM,
        letters="A, B, C, or D",
    )


def internal_knowledge_prompt(query):
    """Figure 9; copied from code/mcq/5_internal=false_o.py."""
    return INTERNAL_KNOWLEDGE_TEMPLATE.format(query=query)


def evidence_prompt(query, evidence):
    """Figure 10, textual evidence. Attach images through the model API."""
    return EVIDENCE_TEMPLATE.format(query=query, evidence=evidence)


def route_mcq_responses(query, responses, choices):
    """New integration of the paper's five-response unanimity rule.

    Uses the ORIGINAL permissive letter parser. This routing function was
    added for release; it is not a recovered historical filtering script.
    """
    if len(responses) != 5:
        raise ValueError("Exactly five MCQ responses are required")
    by_letter = {c["letter"]: c for c in choices}
    letters = [normalize_letter(r) for r in responses]
    if len(set(letters)) != 1 or letters[0] not in by_letter:
        return {"action": "discard", "reason": "inconsistent_or_invalid_mcq"}
    selected = by_letter[letters[0]]
    if selected["tag"] == "none_of_them":
        return {"action": "generate", "prompt": internal_knowledge_prompt(query)}
    return {"action": "keep", "internal_answer": selected["text"],
            "selected_tag": selected["tag"]}
