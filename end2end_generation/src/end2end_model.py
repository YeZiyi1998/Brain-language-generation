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
from Decoder import Decoder, LMFeatures, Hypothesis
sys.path.append('../../language_generation/')
from src.settings import model_name2path, model2hidden
from src.model_utils import Prompt_model
from src.model import Decoding_model
from src.top_model_utils import LanguageModel, Top_model

class End2End_model(Decoding_model):
    def __init__(self, args):
        super().__init__(args)
        self.word_rate_model = joblib.load(f'{args["word_rate_model_path"]}/model.pkl')
        self.args = args
        with open(f'../../data_lm/decoder_vocab.{args["model_name"]}.json', "r") as f:
            decoder_vocab = json.load(f)
        if ('huth' in args['model_name']) is False:
            self.top_model = Top_model(self.model, self.tokenizer, device = self.device, prompt_model = self.prompt_model)
        self.top_model.prompt_model = self.prompt_model
        self.decoder = Decoder()
        self.lm = LanguageModel(self.top_model, decoder_vocab)
        
    # 可行方法
    # (1）brain + text prompt (截断) -> continuation
    # (2) text prompt (截断) + brain + text prompt(截断) -> continuation
    # (3) [brain + text prompt(截断)] +概率合并+ [text prompt] -> continuation
    # (4) text prompt (brain fusion) -> continuation (暂不考虑)
    
    def test(self, test_dataset, file_name=None):
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=1)
        re = {'valid_loss':[], 'content_pred':[], 'content_true':[], 'content_prev':[],'content_pred_token_ids':[],'data_id':[]}
        self.prompt_model.eval()
        if self.args['generation_method'] == 'greedy':
            file_name += '_' + self.args['generation_method']
        
        existing_tokens = self.tokenizer.encode('i')
        
        for content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, content_all, content_all_mask, data_id in tqdm.tqdm(test_dataloader, mininterval=300):
            # estimate word rate
            word_rate = math.ceil(self.word_rate_model.predict([additional_bs.numpy().flatten()])) + 1

            # input construction
            content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask)
            content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)
            existing_tokens_tensor = torch.tensor(existing_tokens[-self.args['prev_mask_len']:], dtype=torch.long)
            
            if len(existing_tokens) > 0:
                content_prev_mask = content_prev_mask.zero_()
                content_prev_mask[0][:len(existing_tokens)] = 1
                content_prev[0][:len(existing_tokens_tensor)] = existing_tokens_tensor
            
            # generate
            all_predicted_tokens = self.prompt_model.generate(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, mode='test')
            all_predicted_tokens[0] = all_predicted_tokens[0][:word_rate]
            existing_tokens += [item.detach().cpu().numpy().tolist() for item in all_predicted_tokens[0]]
            data_id = data_id.numpy().tolist()
            re['content_true'].append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(content_true[0])).replace('<|endoftext|>','').replace('⁇','').replace('</s>','').replace('<unk>','').strip())
            predicted_tokens = all_predicted_tokens[0]
            content_pred_tokens = self.tokenizer.convert_ids_to_tokens(predicted_tokens)
            re['content_pred_token_ids'].append([item.detach().cpu().numpy().tolist() for item in predicted_tokens])
            re['content_pred'].append(self.tokenizer.convert_tokens_to_string(content_pred_tokens))
            re['content_prev'].append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(content_prev[0])).replace('<|endoftext|>','').replace('⁇','').replace('</s>','').replace('<unk>','').strip())
            re['data_id'].append(data_id[0])
            
            if data_id[0] == 20:
                json.dump(re, open(self.args['checkpoint_path']+'/'+file_name+f'.20.json', 'w'))

        if file_name is not None:
            json.dump(re, open(self.args['checkpoint_path']+'/'+file_name+f'.json', 'w'))

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

    def test_beam(self, test_dataset, file_name=None):
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=1)
        self.prompt_model.eval()
        
        re = {'beam_list':[], 'result':[],'data_id':[], 'content_true':[], 'content_pred':[]}
        decoder = self.decoder
        
        print('startting end to end generation:')
        
        for content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, content_all, content_all_mask, data_id in tqdm.tqdm(test_dataloader, mininterval=300):
            # estimate word rate
            word_rate = int(self.word_rate_model.predict([additional_bs.numpy().flatten()]))
            # word_rate = math.ceil(self.word_rate_model.predict([additional_bs.numpy().flatten()]))
            ncontext = min(5, word_rate)
            word_rate = min(10, word_rate)
            
            # input construction
            content_prev_sep = content_prev_sep.expand(self.decoder.beam_width, -1,)
            additional_bs, additional_bs_mask = self.put_bs_into_cuda(additional_bs)
            content_prev_sep = content_prev_sep.to(self.device)
            data_id = data_id.numpy().tolist()[0]
            re['data_id'].append(data_id)
            re['content_true'].append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(content_true[0])).replace('<|endoftext|>','').replace('⁇','').replace('</s>','').replace('<unk>','').strip())
            
            for i in range(word_rate):
                if self.args['input_method'] == 'without_brain':
                    beam_nucs = self.lm.beam_propose(decoder.beam, ncontext, )
                else:
                    beam_nucs = self.lm.beam_propose(decoder.beam, ncontext, additional_bs=additional_bs, additional_bs_mask=additional_bs_mask, content_prev_sep=content_prev_sep)
                for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()): # 对于所有可能的生成的词
                    nuc, logprobs = beam_nucs[c]
                    if len(nuc) < 1: continue
                    extend_words = [hyp.words + [x] for x in nuc] # n_candidates 计算生成这些词之后的情况 如 [i was at a]
                    # trs [1 2 3 4 5 6]; 
                    local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, [None for _ in range(len(logprobs))])] # 所有可能的extensions
                    likelihoods = logprobs
                    decoder.add_extensions(local_extensions, likelihoods, nextensions) # 基于bayes来生成下一个词接到后面
                decoder.extend(verbose = False)
                # re['beam_list'].append([item.words[-20:] for item in decoder.beam])                
            
            if word_rate > 0:
                re['content_pred'].append([item.words[-word_rate:] for item in decoder.beam])
            else:
                re['content_pred'].append([[] for item in decoder.beam])
            
            if data_id % 20 == 0:
                re['result'].append(decoder.beam[0].words)
            
            if data_id == 20 or data_id == 100:
                json.dump(re, open(self.args['checkpoint_path']+'/'+file_name+f'.{data_id}.json', 'w'))
                print('save results with top 20 steps')
                exit()
        
        re['result'].append(decoder.beam[0].words)
        
        if file_name is not None:
            json.dump(re, open(self.args['checkpoint_path']+'/'+file_name+f'.json', 'w'))
