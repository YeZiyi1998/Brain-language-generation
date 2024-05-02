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
sys.path.append('../../language_generation/')
from src.settings import model_name2path, model2hidden
from src.model_utils import Prompt_model
from src.model import Decoding_model

class End2End_model(Decoding_model):
    def __init__(self, args):
        super().__init__(args)
        self.word_rate_model = joblib.load(f'{args["word_rate_model_path"]}/model.pkl')
    
    # 可行方法
    #     (1）brain + text prompt (截断) -> continuation
    # (2) text prompt (截断) + brain + text prompt(截断) -> continuation
    # (3) [brain + text prompt(截断)] +概率合并+ [text prompt] -> continuation
    # (4) text prompt (brain fusion) -> continuation (暂不考虑)
    
    def test(self, test_dataset, file_name=None):
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=1)
        re = {'valid_loss':[], 'content_pred':[], 'content_true':[], 'content_prev':[],'content_pred_token_ids':[],'data_id':[]}
        self.prompt_model.eval()
        if self.args['generation_method'] == 'greedy':
            file_name += '_' + self.args['generation_method']
        
        existing_tokens = []
        
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




