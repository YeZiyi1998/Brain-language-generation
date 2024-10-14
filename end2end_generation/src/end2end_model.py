import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim
import json
import wandb 
import torch.optim.lr_scheduler as lr_scheduler
import random
import joblib
import sys
import math
import sklearn
import numpy as np
from Decoder import Decoder, Hypothesis
sys.path.append('../../language_generation/')
from src.settings import model_name2path, model2hidden
from src.model_utils import Prompt_model
from src.model import Decoding_model
from src.top_model_utils import LanguageModel, Top_model, TokenLanguageModel

def convert_int64_to_int(data):
    if isinstance(data, dict):
        return {k: convert_int64_to_int(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_int64_to_int(v) for v in data]
    elif isinstance(data, np.int64):
        return int(data)
    else:
        return data

class End2End_model(Decoding_model):
    def __init__(self, args):
        super().__init__(args)
        self.word_rate_model = joblib.load(f'{args["word_rate_model_path"]}/model.pkl')
        self.args = args
        if self.args['use_decoder_vocab'] == False:
            decoder_vocab = None
        else:
            decoder_vocab = json.load(open(f'../../data_lm/decoder_vocab.{args["model_name"]}.json', "r"))
        self.top_model = Top_model(self.model, self.tokenizer, device = self.device, prompt_model = self.prompt_model)
        self.top_model.prompt_model = self.prompt_model
        self.decoder = Decoder(beam_width=self.args['beam_width'], extensions=self.args['extensions'])
        if self.args['model_name'] in ['llama-7b','gpt2-xl']: #  or 'gpt' in self.args['model_name'] gpt with old tokenizer?
            self.lm = TokenLanguageModel(self.top_model, decoder_vocab, model_name=self.args['model_name'],  task_name = self.args['task_name'])
            self.token_based = False
        else:
            self.lm = LanguageModel(self.top_model, decoder_vocab)
            self.token_based = True
            
    def put_bs_into_cuda(self, additional_bs):
        additional_bs = additional_bs.expand(self.decoder.beam_width, -1, -1)
        if type(additional_bs) == list:
            for k in range(len(additional_bs)):
                additional_bs[k] = additional_bs[k].to(self.device)
        else:
            additional_bs = additional_bs.to(self.device)
        if self.args['model_name'] in ['llama-7b']:
            additional_bs_mask = torch.ones([additional_bs.shape[0], additional_bs.shape[1]+2+1]).to(self.device)
        else:
            additional_bs_mask = torch.ones([additional_bs.shape[0], additional_bs.shape[1]+2]).to(self.device)
        if self.args['model_name'] in ['llama-7b',]:
            additional_bs = additional_bs.half()
        return additional_bs, additional_bs_mask

    def generate(self, word_rate, decoder, ncontext, additional_bs, additional_bs_mask, content_prev_sep):
        for i in range(word_rate):
            if self.args['input_method'] == 'without_brain':
                beam_nucs = self.lm.beam_propose(decoder.beam, ncontext, gcontext=self.args['gcontext'])
            else:
                beam_nucs = self.lm.beam_propose(decoder.beam, ncontext, gcontext=self.args['gcontext'], additional_bs=additional_bs, additional_bs_mask=additional_bs_mask, content_prev_sep=content_prev_sep)
            for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()): # 对于所有可能的生成的词
                nuc, logprobs = beam_nucs[c]
                if len(nuc) < 1: continue
                extend_words = [hyp.words + [x] for x in nuc] # n_candidates 计算生成这些词之后的情况 如 [i was at a]
                # trs [1 2 3 4 5 6]; 
                local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, [None for _ in range(len(logprobs))])] # 所有可能的extensions
                likelihoods = logprobs + sum(hyp.logprobs[-self.args['ncontext']:])
                # likelihoods = logprobs 
                decoder.add_extensions(local_extensions, likelihoods, nextensions) # 基于bayes来生成下一个词接到后面
            decoder.extend(verbose = False)

    def test_beam(self, test_dataset, file_name=None, print_half_result=False):
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=1)
        self.prompt_model.eval()
        
        re = {'beam_list':[], 'result':[],'data_id':[], 'content_true':[], 'content_pred':[], 'word_rate':[], 'result_ids':[], 'word_rate_float':[], 'content_pred_ids':[]}
        decoder = self.decoder
        
        print('startting end to end generation:')
        
        for content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, content_all, content_all_mask, data_id in tqdm.tqdm(test_dataloader, mininterval=300):
            # estimate word rate
            word_rate_float = float(self.word_rate_model.predict([additional_bs.numpy().flatten()])[0])
            word_rate = int(word_rate_float-self.args['length_penalty'])
            word_rate = max(word_rate, 0)
            
            ncontext = self.args['ncontext']
            
            # input construction
            content_prev_sep = content_prev_sep.expand(self.decoder.beam_width, -1,)
            additional_bs, additional_bs_mask = self.put_bs_into_cuda(additional_bs)
            content_prev_sep = content_prev_sep.to(self.device)
            data_id = data_id.numpy().tolist()[0]
            re['data_id'].append(data_id)
            if len(content_true[0]) > 0:
                re['content_true'].append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(content_true[0])).replace('<|endoftext|>','').replace('⁇','').replace('</s>','').replace('<unk>','').strip())
            else:
                re['content_true'].append('')
            self.generate(word_rate, decoder, ncontext, additional_bs, additional_bs_mask, content_prev_sep)
            
            if word_rate > 0:
                re['content_pred'].append([item.words[-word_rate:] for item in decoder.beam] if self.token_based else [self.tokenizer.decode(item.words[-word_rate:]) for item in decoder.beam])
                re['content_pred_ids'].append([item.words[-word_rate:] for item in decoder.beam])
            else:
                re['content_pred'].append([[] for item in decoder.beam])
                re['content_pred_ids'].append([item.words[-word_rate:]  for item in decoder.beam])
            
            if data_id % 20 == 0:
                re['result'].append(decoder.beam[0].words if self.token_based else self.tokenizer.decode(decoder.beam[0].words))
            
            re['word_rate'].append(word_rate)
            re['word_rate_float'].append(word_rate_float)
            
            if (data_id == 20 or data_id == 100) and print_half_result:
                re['result_ids'].append(decoder.beam[0].words)
                re = convert_int64_to_int(re)
                json.dump(re, open(self.args['checkpoint_path']+'/'+file_name+f'.{data_id}.json', 'w'))
                print(f'save results with top {data_id} steps')
            
            if len(re['content_pred']) > self.args['num_steps']:
                break
            
        re['result'].append(decoder.beam[0].words if self.token_based else self.tokenizer.decode(decoder.beam[0].words) )
        re['result_ids'].append(decoder.beam[0].words)
        
        if self.args['num_steps'] > len(re['content_pred']):
            self.generate(30, decoder, ncontext, additional_bs, additional_bs_mask, content_prev_sep)
            re['next'] = decoder.beam[0].words if self.token_based else self.tokenizer.decode(decoder.beam[0].words) 
        
        if file_name is not None:
            re = convert_int64_to_int(re)
            json.dump(re, open(self.args['checkpoint_path']+'/'+file_name+f'.json', 'w'))
