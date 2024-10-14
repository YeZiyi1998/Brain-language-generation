try:
    from utils_eval import WER, BLEU, METEOR
except:
    from src.utils_eval import WER, BLEU, METEOR
import nltk
import json
import copy
import re
import numpy as np
import argparse

def segment(result, chunk_size=10,checkpoint_path=''):
    if 'huth' not in checkpoint_path:
        result['content_pred'] = [' '.join(result['content_pred'][i:i+chunk_size]) for i in range(0, len(result['content_pred']), chunk_size)]
    else:
        result['content_pred'] = [''.join(result['content_pred'][i:i+chunk_size]) for i in range(0, len(result['content_pred']), chunk_size)]
    result['content_true'] = [' '.join(result['content_true'][i:i+chunk_size]) for i in range(0, len(result['content_true']), chunk_size)]

def split_content_pred_by_results(re,):
    re['content_pred'] = []
    result = re['result'][-1]
    l = 0
    for i in range(len(re['content_pred_old'])):
        re['content_pred'].append(' '.join(result[l:l+len(re['content_pred_old'][i][0])]))
        l += len(re['content_pred_old'][i][0])

def normalize_text(text_from_tokens):
    text_from_tokens = re.sub(r'(\w+)\.(\w+)', r'\1. \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\?(\w+)', r'\1? \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\!(\w+)', r'\1! \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\:(\w+)', r'\1: \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\;(\w+)', r'\1; \2', text_from_tokens)
    return text_from_tokens

def language_evaluate_mask_with_sig(re, metrics, dataset_name='Huth', checkpoint_path=''):
    for mname, metric in metrics.items():
        re[mname] = np.array([metric.score(ref = [re['content_true_tokens'][i]], pred = [re['content_pred_tokens'][i]]) for i in range(len(re['content_pred_tokens']))])
    return re

def load_metric(remove_stopwords):
    metrics = {}
    metrics["WER"] = WER(use_score = True,remove_stopwords=remove_stopwords)
    metrics["BLEU"] = BLEU(n = 1,remove_stopwords=remove_stopwords)
    metrics["METEOR"] = METEOR(remove_stopwords=remove_stopwords)
    # if "BERT" in args.metrics: metrics["BERT"] = BERTSCORE(idf_sents = np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")), rescale = False, score = "recall")
    return metrics

def segment_data(data):
    data = [data[i] + data[i+1] if i + 1 < len(data) else data[i] for i in range(0, len(data), 2) ]
    return data

def segment_data_huth(data, times, cutoffs):
    return [[x for c, x in zip(times, data) if c >= start and c < end] for start, end in cutoffs]

def windows(start_time, end_time, duration, step = 1):
    start_time, end_time = int(start_time), int(end_time)
    half = int(duration / 2)
    return [(center - half, center + half) for center in range(start_time + half, end_time - half + 1, step)]


if __name__ == '__main__':
    metrics = load_metric()
    # evaluate results reported in pdf
    base_path = '../../data_lm/Huth_results.json'
    result = json.load(open(base_path))
    result['content_true_tokens'] = segment_data([item.strip().split() for item in result['reference']])
    for user in ['Huth_1', 'Huth_2', 'Huth_3']:
        result['content_pred_tokens'] = segment_data([item.strip().split() for item in result[user]])
        language_evaluate_mask_with_sig(result, metrics)
        output_str = user + f": bleu_1: {np.mean(result['BLEU']):.4f} wer: {np.mean(result['WER']):.4f} meteor: {np.mean(result['METEOR']):.4f}"
        print(output_str)
    
    # # evaluate permutated results
    # # It seems that when running Huth's code, the signal is permutated, but this is just what we want to test
    # result_permutated = np.load("../../data_lm/wheretheressmoke.npz")
    # words = result_permutated['words']
    # times = result_permutated['times']
    # cutoffs = windows(times[0], times[-1], duration=20, step=20)
    # result_permutated = segment_data_huth(words, times, cutoffs)
    # result['content_pred_tokens'] = result_permutated
    # language_evaluate_mask_with_sig(result, metrics)
    # output_str = 'Huth_2 permutated' + f": bleu_1: {'%.4f' % np.mean(result['BLEU'])} wer: {'%.4f' % np.mean(result['WER'])} meteor: {'%.4f' % np.mean(result['METEOR'])}"
    # print(output_str)
    
    # free evaluating
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-file_name', type=str, required=False)
    # args = parser.parse_args()
    # base_path = f'../../paper/result/{args.file_name}.json'
    # result = json.load(open(base_path))
    # result['content_true_tokens'] = segment_data([item.strip().split() for item in result['reference']])
    # for user in ['result']:
    #     result['content_pred_tokens'] = segment_data([item.strip().split() for item in result[user]])
    #     language_evaluate_mask_with_sig(result, metrics)
    #     output_str = user + f": bleu_1: {'%.4f' % np.mean(result['BLEU'])} wer: {'%.4f' % np.mean(result['WER'])} meteor: {'%.4f' % np.mean(result['METEOR'])}"
    #     print(output_str)
    
    