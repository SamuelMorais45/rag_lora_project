import pandas as pd
from rouge_score import rouge_scorer as rs

def compute_rouge(hypotheses, references):
    scorer = rs.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    records = []
    for h, r in zip(hypotheses, references):
        s = scorer.score(r, h)
        records.append({
            "ROUGE-1 F1": s["rouge1"].fmeasure,
            "ROUGE-2 F1": s["rouge2"].fmeasure,
            "ROUGE-L F1": s["rougeL"].fmeasure,
        })
    return pd.DataFrame(records)