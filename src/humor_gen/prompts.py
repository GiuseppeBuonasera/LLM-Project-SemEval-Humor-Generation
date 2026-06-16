from __future__ import annotations

import json


SYSTEM_PROMPT = (
    "You are an English humor writer. Return exactly one short joke. "
    "Do not add explanations, labels, greetings, alternatives, or meta-commentary."
)


def build_generation_prompt(item: dict[str, str], contexts: list[str] | None = None) -> str:
    context_block = ""
    if contexts:
        bullets = "\n".join(f"- {ctx}" for ctx in contexts)
        context_block = f"\nOptional inspiration examples:\n{bullets}\n"
    if item["input_type"] == "headline":
        return (
            f"{SYSTEM_PROMPT}\n{context_block}\n"
            f"Task: Write one brief joke in English related to this headline:\n"
            f"Headline: {item['headline']}\n\n"
            "Rules: no preface, no 'Sure', no 'Here is a joke', no explanation, one joke only."
        )
    return (
        f"{SYSTEM_PROMPT}\n{context_block}\n"
        "Task: Write one brief joke in English that naturally includes both required words.\n"
        f"Required word 1: {item['word1']}\n"
        f"Required word 2: {item['word2']}\n\n"
        "Rules: include both words exactly or as clear inflected forms, no preface, no explanation, one joke only."
    )


def build_judge_prompt(item: dict[str, str], joke_a: str, joke_b: str) -> str:
    task_input = item["headline"] if item["input_type"] == "headline" else f"{item['word1']} / {item['word2']}"
    schema = {
        "winner": "A or B or tie",
        "reason": "short explanation",
        "scores": {
            "humor": {"A": "1-5", "B": "1-5"},
            "relevance": {"A": "1-5", "B": "1-5"},
            "constraint_satisfaction": {"A": "1-5", "B": "1-5"},
            "originality": {"A": "1-5", "B": "1-5"},
            "fluency": {"A": "1-5", "B": "1-5"},
        },
    }
    return (
        "You are a blind evaluator for an English humor generation task. "
        "You do not know which model wrote either joke. Compare Joke A and Joke B.\n\n"
        f"Input type: {item['input_type']}\n"
        f"Task input: {task_input}\n\n"
        f"Joke A: {joke_a}\n"
        f"Joke B: {joke_b}\n\n"
        "Evaluate humor, relevance, constraint satisfaction, originality, and fluency. "
        "Return only valid JSON with this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def build_preference_prompt(item: dict[str, str]) -> str:
    return build_generation_prompt(item)
