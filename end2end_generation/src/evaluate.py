from utils_eval import WER, BLEU, METEOR
import nltk
import json
import copy
import re
import numpy as np
import argparse
import os

def segment(result, chunk_size=10,checkpoint_path=''):
    if 'huth' not in checkpoint_path:
        result['content_pred'] = [' '.join(result['content_pred'][i:i+chunk_size]) for i in range(0, len(result['content_pred']), chunk_size)]
    else:
        result['content_pred'] = [''.join(result['content_pred'][i:i+chunk_size]) for i in range(0, len(result['content_pred']), chunk_size)]
    result['content_true'] = [' '.join(result['content_true'][i:i+chunk_size]) for i in range(0, len(result['content_true']), chunk_size)]

def split_content_pred_by_results(re):
    re['content_pred'] = []
    result = re['result'][-1]
    if sum([len(item) for item in re['content_pred_old']]) == len(result[-1]):
        return
    l = 0
    bad_i = []
    for i in range(len(re['content_pred_old'])):
        if result[l:l+len(re['content_pred_old'][i][0])] not in re['content_pred_old'][i]:
            bad_i.append(i)
            re['content_pred'].append('')
            continue
        re['content_pred'].append(' '.join(result[l:l+len(re['content_pred_old'][i][0])]))
        l += len(re['content_pred_old'][i][0])
        

def normalize_text(text_from_tokens):
    text_from_tokens = re.sub(r'(\w+)\.(\w+)', r'\1. \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\?(\w+)', r'\1? \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\!(\w+)', r'\1! \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\:(\w+)', r'\1: \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\;(\w+)', r'\1; \2', text_from_tokens)
    return text_from_tokens

def language_evaluate_mask_with_sig(re, metrics, dataset_name='Huth',checkpoint_path=''):
    re['content_pred_old'] = copy.deepcopy(re['content_pred'])
    split_content_pred_by_results(re)
    # preprocess
    for i in range(len(re['content_true'])):
        re['content_true'][i] = re['content_true'][i].replace('<|endoftext|>','').replace('??','').replace('⁇','').replace('</s>','').replace('<unk>','').replace('  ', ' ')
        re['content_pred'][i] = re['content_pred'][i].replace('<|endoftext|>','').replace('??','').replace('⁇','').replace('</s>','').replace('<unk>','').replace('  ', ' ')
    
    segment(re, checkpoint_path=checkpoint_path)
    
    re['content_pred_tokens'] = []
    re['content_true_tokens'] = []
    for i in range(len(re['content_true'])):
        # 可以考虑对比直接 split和nltk.word_tokenize
        re['content_pred_tokens'].append(normalize_text(re['content_pred'][i]).split())
        if dataset_name in ['Huth']:
            re['content_pred_tokens'][-1] = [word.lower() for word in re['content_pred_tokens'][-1]]
        re['content_true_tokens'].append(re['content_true'][i].split())
    
    for mname, metric in metrics.items():
        try:
            re[mname] = np.array([metric.score(ref = [re['content_true_tokens'][i]], pred = [re['content_pred_tokens'][i]]) for i in range(len(re['content_pred']))])
        except:
            # jiayudebug snippet
            inputs = ''
            while inputs != 'continue':
                try:
                    print(eval(inputs))
                except Exception as e:
                    print('error:', e)
                    pass
                inputs = input()
    return re

def load_metric():
    metrics = {}
    metrics["WER"] = WER(use_score = True)
    metrics["BLEU"] = BLEU(n = 1)
    metrics["METEOR"] = METEOR()
    # if "BERT" in args.metrics: metrics["BERT"] = BERTSCORE(idf_sents = np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")), rescale = False, score = "recall")
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default='', type=str, required=True)
    args = parser.parse_args()
    checkpoint_path = args.dir

    for file_name in ['test.json', 'test_permutated.json', 'test_nobrain.json']:
        file_path = f'../results/{checkpoint_path}/{file_name}'
        if os.path.exists(file_path):
            result = json.load(open(file_path))
            metrics = load_metric()
            language_evaluate_mask_with_sig(result, metrics, checkpoint_path)

            output_str = file_path + f" bleu_1: {'%.3f' % np.mean(result['BLEU'])} wer: {'%.3f' % np.mean(result['WER'])} meteor: {'%.3f' % np.mean(result['METEOR'])}"
            print(output_str)

