from src.utils.helpers import key_tokens

def compute_faithfulness(samples, generated):
    scores = []
    for s, g in zip(samples, generated):
        ctx  = key_tokens(s["Instruction"])
        resp = key_tokens(g)
        scores.append(len(ctx & resp) / len(ctx) if ctx else 0.5)
    return scores