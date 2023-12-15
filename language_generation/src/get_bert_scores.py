from embedding import get_api_embedding, get_model_embedding, get_model_embedding_all, get_model_embedding_part
import os
from jupyter_utils import preprocess_re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer
import json
import pickle
import numpy as np
import torch
import tqdm
import nltk

model_name2path = {
                    'vicuna-7b': '/work/czm/LLMs/FastChat/vicuna-7b',
                    'llama-7b': '/home/yzy/.cache/huggingface/hub/decapoda-research--llama-7b-hf.main.5f98eefcc80e437ef68d457ad7bf167c2c6a1348' if os.path.exists('/home/yzy/') else '/data/home/scv6830/.cache/huggingface/hub/decapoda-research--llama-7b-hf.main.5f98eefcc80e437ef68d457ad7bf167c2c6a1348',
                    'gpt2':'/home/yzy/.cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8',
                    'gpt2-xl':'/home/yzy/.cache/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8',
                }
model2hidden = {'gpt2':768,'gpt2-xl':1600,'openai':1536, 'api2d':1536, 'vicuna-7b':4096, 'llama-7b': 4096}

class Dataset:
    def __init__(self):
        self.api_used_num = 0
        self.save_limit = 100
        self.gpt_initialized = False

    def gpt_initialize(self, ):
        if self.gpt_initialized:
            return self.model, self.tokenizer, self.device
        else:
            model_name = self.args['model']
            cuda = self.args['cuda']
            if model_name in ['llama-7b', 'vicuna-7b']:
                self.tokenizer = LlamaTokenizer.from_pretrained(model_name2path[model_name])
                self.model = LlamaForCausalLM.from_pretrained(model_name2path[model_name]) 
            elif 'gpt' in model_name:
                if model_name in ['gpt23']:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(model_name[:-1])
                    self.model = GPT2LMHeadModel.from_pretrained(model_name[:-1])
                else:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(model_name2path[model_name])
                    self.model = GPT2LMHeadModel.from_pretrained(model_name2path[model_name])
            self.device = torch.device('cuda:'+cuda)
            self.model.to(self.device) 
            self.gpt_initialized = True
        return self.model, self.tokenizer, self.device
    
    def word2embedding(self, data):
        # print('generating word embedding..................')
        re = []
        for i in tqdm.tqdm(range(len(data))):
            if data[i] not in self.content2vec.keys():
                if self.args['model'] == 'openai':
                    self.save_word_embedding = True
                    self.content2vec[data[i]] = get_api_embedding(data[i])
                    self.api_used_num += 1
                    if self.api_used_num % self.save_limit == 0:
                        pickle.dump(self.content2vec, open(f'../embedding/content2vec.{self.args["model"]}.{self.dataset_name}.pkl','wb'))
                elif self.args['model'] == 'api2d':
                    self.save_word_embedding = True
                    self.content2vec[data[i]] = get_api_embedding(data[i])
                    self.api_used_num += 1
                    if self.api_used_num % self.save_limit == 0:
                        pickle.dump(self.content2vec, open(f'../embedding/content2vec.{self.args["model"]}.{self.dataset_name}.pkl','wb'))
                if self.args['model'] in ['gpt23']:
                    self.save_word_embedding = True
                    model, tokenizer, device = self.gpt_initialize()
                    word_embedding = get_model_embedding_all(model, tokenizer, data[i], device)
                    self.content2vec[data[i]] = word_embedding
                else:
                    self.save_word_embedding = True
                    model, tokenizer, device = self.gpt_initialize()
                    word_embedding = get_model_embedding(model, tokenizer, data[i], device)
                    self.content2vec[data[i]] = word_embedding
            if 'glm-130b' in self.args['model'] or self.args['model'] in ['gpt23']:
                re.append(np.squeeze(self.content2vec[data[i]][self.args['layer'],:]))
            else:
                re.append(self.content2vec[data[i]])
        return re
    
    def word2embedding_part(self, data_pre, data):
        re = []
        for i in tqdm.tqdm(range(len(data))):
            if data[i] not in self.content2vec.keys():
                self.save_word_embedding = True
                model, tokenizer, device = self.gpt_initialize()
                word_embedding = get_model_embedding_part(model, tokenizer, data_pre[i], data[i], device)
                self.content2vec[data[i]] = word_embedding
            if 'glm-130b' in self.args['model'] or self.args['model'] in ['gpt23']:
                re.append(np.squeeze(self.content2vec[data[i]][self.args['layer'],:]))
            else:
                re.append(self.content2vec[data[i]])
        return re
    
    def word_embedding_initialized(self, pickle_path):
        # 如果有更新就存
        
        if os.path.exists(pickle_path):
            self.content2vec = pickle.load(open(pickle_path, 'rb'))
        else:
            self.content2vec = {'':np.zeros(model2hidden[self.args['model']])}

embedding_tool = Dataset()

def main(model_name, json_file, dataset_name, metric='bert_scores', cuda = '0'):
    global embedding_tool
    pickle_path = f'../embedding/content2vec.{model_name}.{dataset_name}.pkl' if metric == 'bert_scores' else  f'../embedding/content2vec.{model_name}.{dataset_name}.part.pkl'
    embedding_tool.args = {'cuda': cuda}
    embedding_tool.dataset_name = dataset_name
    embedding_tool.args['model'] = model_name
    embedding_tool.word_embedding_initialized(pickle_path)
    embedding_tool.gpt_initialize()
    re = preprocess_re(json_file)
    if metric == 'bert_scores':
        print('starting generate word embedding for ground truth contents')
        embedding_tool.word2embedding(re['content_true'])
        print('starting generate word embedding for predicted contents')
        embedding_tool.word2embedding(re['content_pred'])
        pickle.dump(embedding_tool.content2vec, open(pickle_path,'wb'))
    elif metric == 'bert_scores_part':
        print('starting generate word embedding for ground truth contents')
        embedding_tool.word2embedding_part(re['content_prev'], re['content_true'])
        print('starting generate word embedding for predicted contents')
        embedding_tool.word2embedding_part(re['content_prev'], re['content_pred'])
        pickle.dump(embedding_tool.content2vec, open(pickle_path,'wb'))
        
    
if __name__ == '__main__':
    # model_name = 'gpt2-xl'
    # dataset_name= 'Pereira'
    # prefix = '../results/Pereira_'
    # suffix = '_gpt2-xl_lr1e-5_tid1,2_a0.1f0.25'
    # metric = 'bert_scores'
    # cuda = '4'
    # for u in ['P01','M02', 'M04', 'M07', 'M15']:
    #     json_path = f'{prefix}{u}{suffix}/inference0.5.json'
    #     main(model_name, json.load(open(json_path)),dataset_name, metric, cuda)
    # prefix = '../results/Pereira_'
    # suffix = '_gpt2-xl_lr1e-5_tid1,2_a0.1f0.25_random'
    # for u in ['P01']:
    #     json_path = f'{prefix}{u}{suffix}/inference0.5.json'
    #     main(model_name, json.load(open(json_path)),dataset_name, metric, cuda)
    
    # model_name = 'gpt2-xl'
    # dataset_name= 'Narrative'
    # prefix = '../results/Narrative_'
    # suffix = '_gpt2-xl_lr1e-5_n100'
    # for u in ['lucy','prettymouth','notthefallintact','tunnel']:
    #     json_path = f'{prefix}{u}{suffix}/test.json'
    #     main(model_name, json.load(open(json_path)),dataset_name)
    # prefix = '../results/Narrative_'
    # suffix = '_gpt2-xl_lr1e-5_n100_random'
    # for u in ['lucy','prettymouth','notthefallintact','tunnel']:
    #     json_path = f'{prefix}{u}{suffix}/test.json'
    #     main(model_name, json.load(open(json_path)),dataset_name)
    
    # ds还没跑完
    # model_name = 'gpt2-xl'
    # dataset_name= 'ds003020'
    # prefix = '../results/ds003020_'
    # suffix = '_gpt2-xl_lr1e-5_tid1,2_n100'
    # metric = 'bert_scores_part'
    # cuda = '3'
    # for u in ['1']:
    #     json_path = f'{prefix}{u}{suffix}/test.json'
    #     main(model_name, json.load(open(json_path)),dataset_name, metric)
    # suffix = '_gpt2-xl_lr1e-5_random_tid1,2_n100'
    # for u in ['1']:
    #     json_path = f'{prefix}{u}{suffix}/test.json'
    #     main(model_name, json.load(open(json_path)),dataset_name, metric, cuda)

    # HP可以跑
    model_name = 'gpt2-xl'
    dataset_name= 'HP'
    prefix = '../results/HP_'
    suffix = '_gpt2-xl_lr1e-4'
    cuda = '4'
    for u in ['F','I','J','H','K','L','M','N']:
        json_path = f'{prefix}{u}{suffix}/test.json'
        main(model_name, json.load(open(json_path)),dataset_name,cuda=cuda)
    prefix = '../results/HP_'
    suffix = '_gpt2-xl_lr1e-4_random'
    for u in ['F',]:
        json_path = f'{prefix}{u}{suffix}/test.json'
        main(model_name, json.load(open(json_path)),dataset_name,cuda=cuda)
