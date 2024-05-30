import os
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from modules.bleu2 import compute_bleu
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

def mask_text(result, mask=None):
    if mask is None:
        mask = [len(item[0]) for item in result['content_true_tokens']]
        for i in range(len(result['content_pred'])):
            min_stop = mask[i]
            result['content_pred_tokens'][i] = result['content_pred_tokens'][i][:min_stop]
    else:
        for i in range(len(result['content_pred'])):
            result['content_pred_tokens'][i] = result['content_pred_tokens'][i][:mask]
    return result

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

def detokenize(tokens):
    text_from_tokens = ' '.join([item.strip() for item in tokens if len(item.strip()) > 0])
    text_from_tokens = text_from_tokens.replace("<s>", "").replace("  ", " ").replace(" ,", ",").replace(" .", ".").replace(" '", "'").replace(" ?", "?").replace(" !", "!").replace(" :", ":").replace(" ;", ";").replace(" %", "%")
    return text_from_tokens

def preprocess_re(re, mask=None,dataset_name=None):
    if 'processed' in re.keys() and re['processed']:
        return re
    for i in range(len(re['content_true'])):
        re['content_true'][i] = re['content_true'][i].replace('<|endoftext|>','').replace('??','').replace('⁇','').replace('</s>','').replace('<unk>','')
        re['content_pred'][i] = re['content_pred'][i].replace('<|endoftext|>','').replace('??','').replace('⁇','').replace('</s>','').replace('<unk>','').replace('<s>','')
    re['content_pred_tokens'] = []
    re['content_true_tokens'] = []

    for i in range(len(re['content_true'])):
        re['content_pred_tokens'].append(nltk.word_tokenize(normalize_text(re['content_pred'][i])))
        if dataset_name in ['Huth']:
            re['content_pred_tokens'][-1] = [word.lower() for word in re['content_pred_tokens'][-1]]
        re['content_true_tokens'].append([nltk.word_tokenize(re['content_true'][i])])
    re = mask_text(re, mask)
    re['content_pred'] = []
    for i in range(len(re['content_pred_tokens'])):
        re['content_pred'].append(detokenize(re['content_pred_tokens'][i]))
    re['processed'] = True
    return re

def language_evaluate_mask_with_sig(re, mask=None, dataset_name=None):
    re['content_pred_old'] = copy.deepcopy(re['content_pred'])
    re = preprocess_re(re, mask, dataset_name = dataset_name)
    
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
    
    re['first_match'] = []
    for i in range(len(re['content_pred_tokens'])):
        if len(re['content_pred_tokens'][i]) > 0:
            re['first_match'].append(re['content_true_tokens'][i][0][0]==re['content_pred_tokens'][i][0])
        
    re['rouge_scores'] = {'rouge-1':{'r':[]},'rouge-l':{'r':[]},}
    for item in rouge_scores:
        re['rouge_scores']['rouge-1']['r'].append(item['rouge-1']['r'])
        re['rouge_scores']['rouge-l']['r'].append(item['rouge-l']['r'])
        
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    re['corpus_bleu_score'] = {}
    for weight in weights_list:
        re['corpus_bleu_score'][len(weight)] = []
        for i in range(len(re['content_pred_tokens'])):
            re['corpus_bleu_score'][len(weight)].append(compute_bleu([re['content_true_tokens'][i]], [re['content_pred_tokens'][i]], max_order = len(weight))[0])
    re['wer'] = []
    for i in range(len(re['content_pred_tokens'])):
        re['wer'].append(wer(re['content_true_tokens'][i][0],re['content_pred_tokens'][i]))
    return re

def get_results(path_name, print_log=False, file_name = 'test.json'):
    if os.path.exists(f'{path_name}/{file_name}'):
        result = json.load(open(f'{path_name}/{file_name}'))
    else:
        print(f'path_name {path_name}/{file_name} not exists')
        return None
    if len(result['content_prev']) <= 50:
        print(f'path_name {path_name}/{file_name} has too few samples')
        return None
    result = language_evaluate_mask_with_sig(result, dataset_name='' if 'Huth' not in path_name else 'Huth')
    if print_log:
        output_str = f"corpus_bleu_score_1: {'%.3f' % np.mean(result['corpus_bleu_score'][1])} rouge_1: {'%.3f' % np.mean(result['rouge_scores']['rouge-1']['r'])} rouge_l: {'%.3f' % np.mean(result['rouge_scores']['rouge-l']['r'])} loss: {'%.3f' % np.mean(result['valid_loss'])} wer: {'%.3f' % np.mean(result['wer'])}"
        print(output_str)
    return result

def get_result_dic_mode():
    return {'bert_scores_part':[], 'bert_scores':[], 'valid_loss':[], 'corpus_bleu_score':{1:[],2:[],3:[],4:[]}, 'rouge_scores': {'rouge-1':{'r':[]}, 'rouge-l':{'r':[]}}, 'content_prev':[], 'content_pred':[], 'content_true':[],'content_pred_old':[],'content_prev_tokens_length':[],'first_match':[], 'u':[], 'wer':[]}

def get_iterate_results_split(path_name_list, print_log=False, ):  
    u2result_list = {}    
    for u, item in enumerate(path_name_list):
        path_name = item['path_name'] 
        file_name = item['file_name']
        result = get_results(path_name, print_log=False, file_name=file_name)
        if result is None:
            continue
        u2result_list[u]  = result
    return u2result_list

def multi_add(dic, times = 10):
    new_dic = get_result_dic_mode()
    u_list = set(dic['u'])
    for u in u_list:
        this_u = [idx for idx, item in enumerate(dic['u']) if item == u]
        for j in range(times):
            add2result_dic(new_dic, dic, u, idx_list=this_u,)
    return new_dic

def merge(u2result, user_list = None):
    result = get_result_dic_mode()
    if user_list == None:
        user_list = u2result.keys()
    for u in user_list:
        add2result_dic(result, u2result[u], u)   
    return result 

def get_iterate_results(path_name, print_log = False, ):
    u2result_list = get_iterate_results_split(path_name, print_log=False, )
    result = merge(u2result_list)
    if print_log:
        output_str = f"corpus_bleu_score_1: {np.mean(result['corpus_bleu_score'][1]):.4f} rouge_1: {np.mean(result['rouge_scores']['rouge-1']['r']):.4f} rouge_l: {np.mean(result['rouge_scores']['rouge-l']['r']):.4f} valid_loss: {np.mean(result['valid_loss']):.4f} wer: {np.mean(result['wer']):.4f}"
        print(output_str)
    return result

def show_significance(result1, result2, excel_output=False,metrics = ['corpus_bleu_score_1', 'rouge_1', 'rouge_l', 'valid_loss', 'wer']):
    output_str = f'corpus_bleu_score_1: {get_compare_list(result1["corpus_bleu_score"][1], result2["corpus_bleu_score"][1])}\n'
    output_str += f'rouge_1: {get_compare_list(result1["rouge_scores"]["rouge-1"]["r"], result2["rouge_scores"]["rouge-1"]["r"])}\n'
    output_str += f'rouge_l: {get_compare_list(result1["rouge_scores"]["rouge-l"]["r"], result2["rouge_scores"]["rouge-l"]["r"])}\n'
    if 'valid_loss' in metrics:
        output_str += f'valid_loss: {get_compare_list(result1["valid_loss"], result2["valid_loss"])}\n'
    output_str += f'wer: {get_compare_list(result1["wer"], result2["wer"])}'
    print(output_str)
    if 'valid_loss' in metrics:
        pairwise_list = [compare(np.array(result1['valid_loss'][idx]), np.array(result2['valid_loss'][idx])) for idx in range(len(result1['valid_loss']))]
        print(f"pairwise accuracy:  {np.sum(pairwise_list)/len(result1['valid_loss']):.4f}",)
    if excel_output:
        for result in [result1, result2]:
            print(f'{np.mean(result["corpus_bleu_score"][1]):.4f},{np.mean(result["rouge_scores"]["rouge-1"]["r"]):.4f},{np.mean(result["rouge_scores"]["rouge-l"]["r"]):.4f},{np.mean(result["valid_loss"]):.4f},{np.mean(result["wer"]):.4f}')

def add2result_dic(dic1, dic2, u,idx_list=None,):
    if idx_list == None:
        idx_list = [0, len(dic2['content_pred'])]
    else:
        idx_list = [min(idx_list), max(idx_list)+1]
    for i in range(1,5):
        dic1['corpus_bleu_score'][i] += dic2['corpus_bleu_score'][i][idx_list[0]:idx_list[1]] if i in dic2['corpus_bleu_score'].keys() else dic2['corpus_bleu_score'][str(i)][idx_list[0]:idx_list[1]]
    for k1 in ['rouge-1', 'rouge-l']:
        for k2 in ['r']:
            dic1['rouge_scores'][k1][k2] += dic2['rouge_scores'][k1][k2][idx_list[0]:idx_list[1]]
    for k in ['bert_scores', 'valid_loss', 'bert_scores_part', 'content_true', 'content_pred', 'content_prev', 'content_pred_old', 'content_prev_tokens_length','first_match','wer']:
        if k in dic2.keys():
            dic1[k] += dic2[k][idx_list[0]:idx_list[1]]
    for i in range(idx_list[1]-idx_list[0]):
        dic1['u'].append(u)

def get_compare_list(v1, v2):
    mean_v1 = np.mean(v1)
    mean_v2 = np.mean(v2)
    alternative = 'less'if mean_v1 < mean_v2 else 'greater'
    try:
        u_statistic, p_value_u = stats.wilcoxon(v1, v2, alternative=alternative)
    except:
        p_value_u = 1.0
    t_statistic, p_value_t = stats.ttest_rel(v1, v2,)
    return f'{mean_v1:.4f},{mean_v2:.4f}:\tu-test:{p_value_u:.4f},t-test:{p_value_t:.4f}'

def compare(a,b):
    if a==b:
        return 0.5
    return 1 if a<b else 0

def is_only_dot_space(text):
    pattern = r'^[.\s]+$'
    match = re.match(pattern, text)
    if match:
        return True
    else:
        return False

if __name__ == '__main__':
    result_path = 'example'
    # comparing BrainLLM and PerBrainLLM
    base_path = '../results/'
    model_dir_list = [{'path_name':base_path + result_path, 'file_name':'test.json'}]
    control_dir_list = [{'path_name':base_path + result_path, 'file_name':'test_permutated.json'}]
    model_result = get_iterate_results(model_dir_list, print_log=True)
    control_result = get_iterate_results(control_dir_list)
    if len(model_result['content_prev']) < len(control_result['content_prev']):
        if len(model_result['content_prev']) * 10 == len(control_result['content_prev']):
            model_result = multi_add(model_result)
        else:
            print("Error: length of data samples in the proposed model and the control model doesn't not match")
    
    show_significance(model_result, control_result)
    
    # comparing BrainLLM and StdLLM
    base_path = '../results/'
    model_dir_list = [{'path_name':base_path + result_path, 'file_name':'test.json'}]
    control_dir_list = [{'path_name':base_path + result_path, 'file_name':'test_nobrain.json'}]
    model_result = get_iterate_results(model_dir_list)
    control_result = get_iterate_results(control_dir_list)
    if len(model_result['content_prev']) < len(control_result['content_prev']):
        if len(model_result['content_prev']) * 10 == len(control_result['content_prev']):
            model_result = multi_add(model_result)
        else:
            print("Error: length of data samples in the proposed model and the control model doesn't not match")
    
    show_significance(model_result, control_result)
