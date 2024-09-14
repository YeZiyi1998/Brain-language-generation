from jiwer import wer
import numpy as np
# import evaluate
from nltk.translate import meteor_score
try:
    from download_metrics.bleu2 import compute_bleu
except:
    from end2end_generation.src.download_metrics.bleu2 import compute_bleu
from nltk import word_tokenize
from nltk.corpus import stopwords
from bert_score import BERTScorer

stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    global stop_words
    if type(tokens[0]) == list:
        re = []
        for i in range(len(tokens)):
            re.append([word for word in tokens[i] if word.lower() not in stop_words])
        return re
    else:
        return [word for word in tokens if word.lower() not in stop_words]

"""
WER
"""
class WER(object):
    def __init__(self, use_score = True, remove_stopwords=False):
        self.use_score = use_score
        self.remove_stopwords = remove_stopwords
    
    def score(self, ref, pred):
        if self.remove_stopwords:
            ref, pred = remove_stopwords(ref), remove_stopwords(pred)
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
    def __init__(self, n = 4, remove_stopwords=False):
        # self.metric = evaluate.load("bleu", keep_in_memory=True, trust_remote_code=True)
        self.n = n
        self.remove_stopwords=remove_stopwords
    
    def score(self, ref, pred):
        if self.remove_stopwords:
            ref, pred = remove_stopwords(ref), remove_stopwords(pred)
        scores = []
        for i in range(len(ref)):
            scores.append(compute_bleu([[ref[i]]], [pred[i]], max_order=self.n)[0])
        return np.array(scores)
    
class METEOR(object):
    def __init__(self, remove_stopwords=False):
        self.remove_stopwords = remove_stopwords

    def _compute(self, predictions, references, alpha=0.9, beta=3, gamma=0.5):
        scores = [
            meteor_score.single_meteor_score(
                word_tokenize(ref), word_tokenize(pred), alpha=alpha, beta=beta, gamma=gamma
            )
            for ref, pred in zip(references, predictions)
        ]
        return scores
    
    def score(self, ref, pred):
        if self.remove_stopwords:
            ref, pred = remove_stopwords(ref), remove_stopwords(pred)
        results = []
        ref_strings = [' '.join(x) for x in ref]
        pred_strings = [' '.join(x) for x in pred]
        return self._compute(ref_strings, pred_strings)
    
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
    