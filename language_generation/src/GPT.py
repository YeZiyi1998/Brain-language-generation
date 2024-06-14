import torch
import numpy as np
from transformers import AutoModelForCausalLM
from torch.nn.functional import softmax
import copy

class GPT_Tokenizer():
    def __init__(self, gpt):
        self.gpt = gpt
        self.vocab = self.gpt.vocab
        self.eos_token = '<unk>'
        self.pad_token = self.eos_token
        self.word2id = self.gpt.word2id
        self.pad_token_id = self.word2id[self.pad_token]
        self.eos_token_id = self.word2id[self.eos_token]
        self.id2word = dict((v,k) for k,v in self.word2id.items())
        
    def encode_plus(self, words, max_length=None, truncation=True, padding='max_length', return_tensors=None, add_special_tokens=None):
        if type(words) is str:
            words = words.split()
        if max_length is None:
            max_length = len(words)
        # 将单词转换为id
        input_ids = [self.word2id[x] if x in self.word2id else self.pad_token_id for x in words]
        # 创建attention_mask
        attention_mask = [1] * len(input_ids)
        # 截断和padding
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        elif padding == 'max_length' and len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        # 将input_ids和attention_mask转换为tensor
        input_ids = torch.tensor([input_ids])
        attention_mask = torch.tensor([attention_mask])
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def encode(self, word, add_special_tokens=None):
        return [self.word2id[item] for item in word]

    def add_tokens(self, new_tokens):
        for new_token in new_tokens:
            self.word2id[new_token] = len(self.vocab)
            self.vocab.append(new_token)
        self.id2word = dict((v,k) for k,v in self.word2id.items())
        
    def convert_tokens_to_ids(self, token):
        if type(token) is str:
            return self.word2id[token]
        else:
            return [self.word2id[item] for item in token]

    def get_vocab(self, ):
        return self.word2id

    def __len__(self, ):
        return len(self.vocab)
    
    def convert_ids_to_tokens(self, input_ids):
        return [self.id2word[item.detach().cpu().numpy().tolist()] for item in input_ids]
        
    def convert_tokens_to_string(self, input_tokens):
        return ' '.join(input_tokens)

class GPT():    
    """wrapper for https://huggingface.co/openai-gpt
    """
    def __init__(self, path, vocab, device = 'cpu', prompt_model=None):  
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(path).eval().to(self.device)
        self.vocab = vocab
        self.word2id = {w : i for i, w in enumerate(self.vocab)}
        self.UNK_ID = self.word2id['<unk>']
        self.prompt_model = prompt_model

    def resize_token_embeddings(self, len_token):
        self.model.resize_token_embeddings(len_token)

    def state_dict(self,):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)
    
    def encode(self, words):
        """map from words to ids
        """
        return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]
        