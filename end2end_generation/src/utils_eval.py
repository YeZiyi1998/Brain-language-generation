from jiwer import wer
import numpy as np
# import evaluate
from nltk.translate import meteor_score
from download_metrics.bleu2 import compute_bleu
from nltk import word_tokenize
from bert_score import BERTScorer
"""
WER
"""
class WER(object):
    def __init__(self, use_score = True):
        self.use_score = use_score
    
    def score(self, ref, pred):
        ref_strings = [' '.join(x) for x in ref]
        pred_strings = [' '.join(x) for x in pred]
        scores = []
        for ref_seg, pred_seg in zip(ref_strings, pred_strings):
            scores.append(wer(ref_seg, pred_seg))
        return np.array(scores)
    
"""
BLEU (https://aclanthology.org/P02-1040.pdf)
"""
class BLEU(object):
    def __init__(self, n = 4):
        # self.metric = evaluate.load("bleu", keep_in_memory=True, trust_remote_code=True)
        self.n = n
    
    def score(self, ref, pred):
        scores = []
        for i in range(len(ref)):
            scores.append(compute_bleu([[ref[i]]], [pred[i]], max_order=self.n)[0])
        return np.array(scores)
    
class METEOR(object):
    def _compute(self, predictions, references, alpha=0.9, beta=3, gamma=0.5):
        scores = [
            meteor_score.single_meteor_score(
                word_tokenize(ref), word_tokenize(pred), alpha=alpha, beta=beta, gamma=gamma
            )
            for ref, pred in zip(references, predictions)
        ]

        return scores
    
    def score(self, ref, pred):
        results = []
        ref_strings = [' '.join(x) for x in ref]
        pred_strings = [' '.join(x) for x in pred]
        return self._compute(ref_strings, pred_strings)
        
"""
BERTScore (https://arxiv.org/abs/1904.09675)
"""
class BERTSCORE(object):
    def __init__(self, idf_sents=None, rescale = True, score = "f"):
        self.metric = BERTScorer(lang = "en", rescale_with_baseline = rescale, idf = (idf_sents is not None), idf_sents = idf_sents)
        if score == "precision": self.score_id = 0
        elif score == "recall": self.score_id = 1
        else: self.score_id = 2

    def score(self, ref, pred):
        ref_strings = [' '.join(x) for x in ref]
        pred_strings = [' '.join(x) for x in pred]
        return self.metric.score(cands = pred_strings, refs = ref_strings)[self.score_id].numpy()
    
    
class BERTSCORE(object):
    def __init__(self, idf_sents=None, rescale = False, score = "recall"):
        self.metric = BERTScorer(lang = "en", rescale_with_baseline = rescale, idf = (idf_sents is not None), idf_sents = idf_sents)
        if score == "precision": self.score_id = 0
        elif score == "recall": self.score_id = 1
        else: self.score_id = 2

    def score(self, ref, pred):
        ref_strings = [" ".join(x) for x in ref]
        pred_strings = [" ".join(x) for x in pred]
        return self.metric.score(cands = pred_strings, refs = ref_strings)[self.score_id].numpy()
    