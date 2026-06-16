from src.utils.helpers import key_tokens

def compute_relevance(samples, generated):
    scores = []
    for s, g in zip(samples, generated):
        q, r = key_tokens(s["Instruction"]), key_tokens(g)
        if not q and not r:
            scores.append(0.5); continue
        jaccard  = len(q & r) / len(q | r) if q | r else 0
        coverage = len(q & r) / len(q) if q else 0
        scores.append((jaccard + coverage) / 2.0)
    return scores