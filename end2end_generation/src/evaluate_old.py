
import os
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate import meteor_score
from rouge import Rouge
import re
import copy
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
rouge = Rouge()
chencherry = SmoothingFunction()

def wer(reference_words, hypothesis_words):
    if type(reference_words) == str:
        reference_words = reference_words.split()
    if type(reference_words) == str:
        hypothesis_words = hypothesis_words.split()
    # Initialize a matrix of size (len(reference) + 1) x (len(hypothesis) + 1)
    d = np.zeros((len(reference_words)+1, len(hypothesis_words)+1), dtype=np.uint8)
    # Fill the first column with 0, 1, 2, ..., len(reference)
    for i in range(len(reference_words)+1):
        d[i, 0] = i
    # Fill the first row with 0, 1, 2, ..., len(hypothesis)
    for j in range(len(hypothesis_words)+1):
        d[0, j] = j
    # Populate the matrix
    for i in range(1, len(reference_words)+1):
        for j in range(1, len(hypothesis_words)+1):
            substitute = d[i-1, j-1] + (reference_words[i-1] != hypothesis_words[j-1])
            insert = d[i, j-1] + 1
            delete = d[i-1, j] + 1
            d[i, j] = min(substitute, insert, delete)
    # The bottom-right element is the Levenshtein distance
    edit_distance = d[len(reference_words), len(hypothesis_words)]
    # WER is Levenshtein distance normalized by the length of the reference
    return edit_distance / len(reference_words)

def normalize_text(text_from_tokens):
    text_from_tokens = re.sub(r'(\w+)\.(\w+)', r'\1. \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\?(\w+)', r'\1? \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\!(\w+)', r'\1! \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\:(\w+)', r'\1: \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\;(\w+)', r'\1; \2', text_from_tokens)
    return text_from_tokens

vocab = json.load(open('../../data_lm/decoder_vocab.json'))
def clean_text(word_list):
    return [word for word in word_list if word in vocab]

def detokenize(tokens):
    text_from_tokens = ' '.join([item.strip() for item in tokens if len(item.strip()) > 0])
    text_from_tokens = text_from_tokens.replace("  ", " ").replace(" ,", ",").replace(" .", ".").replace(" '", "'").replace(" ?", "?").replace(" !", "!").replace(" :", ":").replace(" ;", ";").replace(" %", "%")
    return text_from_tokens

def preprocess_re(re, mask=None,dataset_name=None):
    if 'processed' in re.keys() and re['processed']:
        return re
    for i in range(len(re['content_true'])):
        re['content_true'][i] = re['content_true'][i].replace('<|endoftext|>','').replace('??','').replace('⁇','').replace('</s>','').replace('<unk>','').replace('  ', ' ')
    re['content_pred_tokens'] = []
    re['content_true_tokens'] = []

    for i in range(len(re['content_true'])):
        re['content_pred_tokens'].append(nltk.word_tokenize(normalize_text(re['content_pred'][i])))
        if dataset_name in ['Huth']:
            re['content_pred_tokens'][-1] = [word.lower() for word in re['content_pred_tokens'][-1]]
        re['content_true_tokens'].append([nltk.word_tokenize(re['content_true'][i])])

    re['content_pred'] = []
    re['content_true'] = []
    re['content_true_tokens'] = [item[0] for item in re['content_true_tokens']]
    for i in range(len(re['content_pred_tokens'])):
        re['content_pred_tokens'][i] = clean_text(re['content_pred_tokens'][i])
        re['content_pred'].append(detokenize(re['content_pred_tokens'][i]))
        re['content_true_tokens'][i] = clean_text(re['content_true_tokens'][i])
        re['content_true'].append(detokenize(re['content_true_tokens'][i]))
    re['processed'] = True
    return re

def segment(re, chunk_size=10):
    re['content_pred'] = [' '.join(re['content_pred'][i:i+chunk_size]) for i in range(0, len(re['content_pred']), chunk_size)]
    re['content_true'] = [' '.join(re['content_true'][i:i+chunk_size]) for i in range(0, len(re['content_true']), chunk_size)]
    re['content_pred_tokens'] = [sum(re['content_pred_tokens'][i:i+chunk_size], []) for i in range(0, len(re['content_pred_tokens']), chunk_size)]
    re['content_true_tokens'] = [sum(re['content_true_tokens'][i:i+chunk_size], []) for i in range(0, len(re['content_true_tokens']), chunk_size)]

def split_content_pred_by_results(re,):
    re['content_pred'] = []
    result = re['result'][-1]
    l = 0
    for i in range(len(re['content_pred_old'])):
        re['content_pred'].append(' '.join(result[l:l+len(re['content_pred_old'][i][0])]))
        l += len(re['content_pred_old'][i][0])

def language_evaluate_mask_with_sig(re, mask=None, dataset_name=None):
    re['content_pred_old'] = copy.deepcopy(re['content_pred'])
    split_content_pred_by_results(re)
    re = preprocess_re(re, mask, dataset_name = dataset_name)
    segment(re)
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(re['content_pred'],re['content_true'], avg = False)
    except:
        try:
            filtered_idx = [idx for idx in range(len(re['content_pred'])) if is_only_dot_space(re['content_true'][idx])==False]
            selected_idx = [idx for idx in filtered_idx if is_only_dot_space(re['content_pred'][idx])==False and len(re['content_pred'][idx]) > 0]
            rouge_scores = rouge.get_scores([re['content_pred'][idx] for idx in selected_idx],[re['content_true'][idx] for idx in selected_idx], avg = False)
            for _ in range(len(filtered_idx) - len(selected_idx)):
                rouge_scores.append({'rouge-1':{'r':0},'rouge-l':{'r':0}})
        except:
            for idx in range(len(re['content_pred'])):
                rouge_scores = rouge.get_scores([re['content_pred'][idx]],[re['content_true'][idx]], avg = False)
        
    re['rouge_scores'] = {'rouge-1':{'r':[]},'rouge-l':{'r':[]},}
    for item in rouge_scores:
        re['rouge_scores']['rouge-1']['r'].append(item['rouge-1']['r'])
        re['rouge_scores']['rouge-l']['r'].append(item['rouge-l']['r'])
        
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    re['corpus_bleu_score'] = {}
    for weight in weights_list:
        re['corpus_bleu_score'][len(weight)] = []
        for i in range(len(re['content_pred_tokens'])):
            re['corpus_bleu_score'][len(weight)].append(corpus_bleu([re['content_true_tokens'][i]], [re['content_pred_tokens'][i]], weights = weight, smoothing_function = chencherry.method1))
    re['wer'] = []
    re['meteor'] = []
    for i in range(len(re['content_pred_tokens'])):
        re['wer'].append(wer(re['content_true_tokens'][i],re['content_pred_tokens'][i]))
        re['meteor'].append(meteor_score.meteor_score([re['content_true_tokens'][i]], re['content_pred_tokens'][i]))
     
    return re

def is_only_dot_space(text):
    pattern = r'^[.\s]+$'
    match = re.match(pattern, text)
    if match:
        return True
    else:
        return False

if __name__ == '__main__':
    checkpoint_path = 'Huth_1_huth'
    result = json.load(open(f'../results/{checkpoint_path}/test_nobrain2.json'))
    language_evaluate_mask_with_sig(result)

    output_str = f"corpus_bleu_score_1: {'%.3f' % np.mean(result['corpus_bleu_score'][1])} rouge_1: {'%.3f' % np.mean(result['rouge_scores']['rouge-1']['r'])} rouge_l: {'%.3f' % np.mean(result['rouge_scores']['rouge-l']['r'])} wer: {'%.3f' % np.mean(result['wer'])} meteor: {'%.3f' % np.mean(result['meteor'])}"
    print(output_str)


