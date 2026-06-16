import json, re
from pathlib import Path

def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def save_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_prompt(instruction: str, output: str = "") -> str:
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt + output if output else prompt

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","have","has",
    "had","do","does","did","will","to","of","in","on","at","for",
    "with","by","or","and","but","if","o","a","e","de","do","da",
    "em","no","na","para","com","por","que","um","uma","os","as",
}

def key_tokens(text: str) -> set:
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.lower())
    return {t for t in tokens if (t.isdigit() or len(t) >= 4) and t not in STOPWORDS}