from utils_eval import WER, BLEU, METEOR
import nltk
import json
import copy
import re
import numpy as np
import argparse
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer

def segment(result, chunk_size=10,checkpoint_path=''):
    if 'huth' not in checkpoint_path:
        result['content_pred'] = [' '.join(result['content_pred'][i:i+chunk_size]) for i in range(0, len(result['content_pred']), chunk_size)]
    else:
        result['content_pred'] = [''.join(result['content_pred'][i:i+chunk_size]) for i in range(0, len(result['content_pred']), chunk_size)]
    result['content_true'] = [' '.join(result['content_true'][i:i+chunk_size]) for i in range(0, len(result['content_true']), chunk_size)]

def split_content_pred_by_results(re):
    re['content_pred'] = []
    result = re['result'][-1]
    l = 0
    bad_i = []
    for i in range(len(re['content_pred_old'])):
        if result[l:l+len(re['content_pred_old'][i][0])] not in re['content_pred_old'][i]:
            bad_i.append(i)
            re['content_pred'].append('')
            continue
        re['content_pred'].append(' '.join(result[l:l+len(re['content_pred_old'][i][0])]))
        l += len(re['content_pred_old'][i][0])
    if len(bad_i) > 0:
        print('bad_i', bad_i)

tokenizer = None

def split_content_pred_by_results2(re, checkpoint_path):
    global tokenizer
    if tokenizer is None:
        if 'gpt2' in checkpoint_path:
            tokenizer = AutoTokenizer.from_pretrained('/home/bingxing2/home/scx7140/.cache/huggingface/hub/models--gpt2-large/snapshots/97935fc1a406f447320c3db70fe9e9875dca2595')
        elif 'llama-7b' in checkpoint_path:
            tokenizer = AutoTokenizer.from_pretrained('/home/bingxing2/home/scx7140/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852')  
    re['content_pred'] = []
    result = re['result_ids'][-1]
    l = 0
    for i in range(len(re['word_rate'])):
        if re['word_rate'][i] < 0:
            re['content_pred'].append('')
            if i + 1 < len(re['word_rate']):
                re['word_rate'][i+1] += re['word_rate'][i]
        elif re['word_rate'][i] == 0:
            re['content_pred'].append('')
        else:
            tmp_result = result[l:l+re['word_rate'][i]]
            k = 0
            while l+re['word_rate'][i]+k < len(result) and len(tokenizer.decode(result[l:l+re['word_rate'][i]+k]).split()) == len(tokenizer.decode(result[l:l+re['word_rate'][i]+k+1]).split()):
                k += 1
            re['content_pred'].append(tokenizer.decode(result[l:l+re['word_rate'][i]+k]))
            l += re['word_rate'][i]+k
            if i + 1 < len(re['word_rate']):
                re['word_rate'][i+1] -= k
        
def normalize_text(text_from_tokens):
    text_from_tokens = re.sub(r'(\w+)\.(\w+)', r'\1. \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\?(\w+)', r'\1? \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\!(\w+)', r'\1! \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\:(\w+)', r'\1: \2', text_from_tokens)
    text_from_tokens = re.sub(r'(\w+)\;(\w+)', r'\1; \2', text_from_tokens)
    return text_from_tokens

def language_evaluate_mask_with_sig(re, metrics, dataset_name='Huth',checkpoint_path=''):
    re['content_pred_old'] = copy.deepcopy(re['content_pred'])
    if 'huth' not in checkpoint_path:
        split_content_pred_by_results2(re, checkpoint_path)
    else:
        split_content_pred_by_results(re, )
    # preprocess
    for i in range(len(re['content_true'])):
        re['content_true'][i] = re['content_true'][i].replace('<|endoftext|>','').replace('??','').replace('⁇','').replace('</s>','').replace('<unk>','').replace('  ', ' ').replace('\n', ' ')
        re['content_pred'][i] = re['content_pred'][i].replace('<|endoftext|>','').replace('??','').replace('⁇','').replace('</s>','').replace('<unk>','').replace('  ', ' ').replace('\n', ' ').replace('<s>', '')
    
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

    # for file_name in ['output.n10.json', 'output.n5.json']:
    for file_name in ['test.json']:
        file_path = f'../results/{args.dir}/{file_name}'
        if os.path.exists(file_path):
            result = json.load(open(file_path))
            # if type(result['content_pred'][0][0]) is str:
            #     for i in range(len(result['content_pred'])):
            #         for k in range(5):
            #             result['content_pred'][i][k] = result['content_pred'][i][k].strip().split()
            #     for i in range(len(result['result'])):
            #         result['result'][i] = result['result'][i].strip().split()
            metrics = load_metric()
            language_evaluate_mask_with_sig(result, metrics, checkpoint_path=args.dir)

            output_str = file_path + f" bleu_1: {'%.3f' % np.mean(result['BLEU'])} wer: {'%.3f' % np.mean(result['WER'])} meteor: {'%.3f' % np.mean(result['METEOR'])}"
            print(output_str)

