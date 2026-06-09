import re, numpy as np

def _features(text):
    return {
        "numbered" : bool(re.search(r"^\d+\.\s", text, re.MULTILINE)),
        "bullets"  : bool(re.search(r"^[\-•]\s",  text, re.MULTILINE)),
        "technical": bool(re.search(
            r"\d+[,.]?\d*\s*(psi|mph|kg|km|liter|L|mL|mg|ms|MHz|GB|MB|%)",
            text, re.IGNORECASE)),
        "sections" : bool(re.search(r"^[A-Z][^\n]+:\s*$", text, re.MULTILINE)),
        "steps"    : len(re.findall(r"^\d+\.\s", text, re.MULTILINE)),
    }

def compute_plan_adherence(samples, generated):
    scores = []
    for s, g in zip(samples, generated):
        rf, gf = _features(s["Output"]), _features(g)
        checks = []
        for feat in ("numbered","bullets","technical","sections"):
            if rf[feat]:
                checks.append(1.0 if gf[feat] else 0.0)
        if rf["steps"] > 0:
            checks.append(min(gf["steps"], rf["steps"]) / max(gf["steps"], rf["steps"], 1))
        scores.append(float(np.mean(checks)) if checks else 0.5)
    return scores