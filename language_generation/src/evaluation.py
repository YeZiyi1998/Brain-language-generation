from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk

def language_evaluate(re):
    """ calculate rouge score """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(re['content_pred'],re['content_true'], avg = True)
    re['rouge_scores'] = rouge_scores

    re['content_pred_tokens'] = []
    re['content_true_tokens'] = []

    for i in range(len(re['content_pred'])):
        re['content_pred_tokens'].append(nltk.word_tokenize(re['content_pred'][i]))
        re['content_true_tokens'].append([nltk.word_tokenize(re['content_true'][i])])
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    re['corpus_bleu_score'] = {}
    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(re['content_true_tokens'], re['content_pred_tokens'], weights = weight)
        re['corpus_bleu_score'][len(weight)] = corpus_bleu_score
    return re

def language_evaluate_mask(re, mask=None):
    """ calculate rouge score """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(re['content_pred'],re['content_true'], avg = True)
    re['rouge_scores'] = rouge_scores
    
    re['content_pred_tokens'] = []
    re['content_true_tokens'] = []

    for i in range(len(re['content_pred'])):
        re['content_pred_tokens'].append(nltk.word_tokenize(re['content_pred'][i]))
        re['content_true_tokens'].append([nltk.word_tokenize(re['content_true'][i])])
    # 要不改成真实的有多少tokens就多少？
    if mask is None:
        mask = int(np.mean([len(item[0]) for item in re['content_true_tokens']])) + 1
    for i in range(len(re['content_pred'])):
        re['content_pred_tokens'][i] = re['content_pred_tokens'][i][:mask]
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    re['corpus_bleu_score'] = {}
    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(re['content_true_tokens'], re['content_pred_tokens'], weights = weight)
        re['corpus_bleu_score'][len(weight)] = corpus_bleu_score
    return re
