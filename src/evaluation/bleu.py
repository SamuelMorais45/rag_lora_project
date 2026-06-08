from sacrebleu.metrics import BLEU as SacreBLEU

def compute_bleu(hypotheses, references):
    bleu = SacreBLEU(effective_order=True)
    corpus = bleu.corpus_score(hypotheses, [references])
    per_sample = [bleu.sentence_score(h, [r]).score
                  for h, r in zip(hypotheses, references)]
    return {
        "bleu_corpus": corpus.score,
        "1-gram": corpus.precisions[0], "2-gram": corpus.precisions[1],
        "3-gram": corpus.precisions[2], "4-gram": corpus.precisions[3],
        "bp": corpus.bp,
    }, per_sample