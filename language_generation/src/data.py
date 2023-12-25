import pickle
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import gc
import json
import copy
from sklearn.preprocessing import StandardScaler

class Splited_FMRI_dataset(Dataset):
    def __init__(self,inputs,most_epoch=-1, args = None):
        self.device = torch.device(f"cuda:{args['cuda']}")
        self.inputs = inputs
        self.most_epoch = most_epoch
        self.args = args
    def __len__(self):
        if self.most_epoch > -1:
            return min(self.most_epoch, len(self.inputs))
        return len(self.inputs)
    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
                input_sample['content_prev'], 
                input_sample['additional_bs'],
                input_sample['content_prev_sep'],
                input_sample['content_true'],
                input_sample['content_prev_mask'],
                input_sample['content_true_mask'],
                input_sample['content_all'],
                input_sample['content_all_mask'],
                input_sample['id']
            )

class FMRI_dataset():
    def add_context_noise(self, input_dataset):
        self.mean_embedding = torch.mean(self.decoding_model.model.transformer.wte.weight, axis=0)
        self.decoding_model.tokenizer.add_tokens('[UNK]')
        new_token_id = self.decoding_model.tokenizer.convert_tokens_to_ids('[UNK]')
        self.decoding_model.model.resize_token_embeddings(len(self.decoding_model.tokenizer))
        self.decoding_model.model.transformer.wte.weight[new_token_id].data = self.mean_embedding.detach()
        def mask_half_tokens(text, new_token_id, noise_ratio):
            num_tokens = len(text)
            num_masked_tokens = int(num_tokens * noise_ratio)
            mask_indices = random.sample(range(num_tokens), num_masked_tokens)
            for i, token in enumerate(text):
                if i in mask_indices:
                    text[i] = new_token_id

        for item in input_dataset:
            mask_half_tokens(item['content_prev'], new_token_id, self.args['noise_ratio'])
        return input_dataset
    
    def pack_info(self, content_prev, additional_bs, content_true, trail_id,id):
        if self.args['model_name'] in ['llama-7b',]:
            self.add_special_tokens = True
            content_all = self.tokenizer.encode_plus(content_prev+' '+content_true, max_length=64, truncation=True, return_tensors='pt', add_special_tokens = self.add_special_tokens, padding='max_length',)
            content_all['input_ids'] = content_all['input_ids'][:,1:]
            content_all['attention_mask'] = content_all['attention_mask'][:,1:]
            content_true = self.tokenizer.encode_plus(content_true.strip(), max_length=32, truncation=True, return_tensors='pt',padding='max_length',)
            content_true['input_ids'] = content_true['input_ids'][:,1:]
            content_true['attention_mask'] = content_true['attention_mask'][:,1:]
            content_prev = self.tokenizer.encode_plus(content_prev, max_length=32, truncation=True, return_tensors='pt', add_special_tokens = self.add_special_tokens, padding='max_length', )
            content_prev['input_ids'] = content_prev['input_ids'][:,1:]
            content_prev['attention_mask'] = content_prev['attention_mask'][:,1:]
        else:
            content_all = self.tokenizer.encode_plus(content_prev+' '+content_true, max_length=64, truncation=True, return_tensors='pt', add_special_tokens = self.add_special_tokens, padding='max_length')
            content_true = self.tokenizer.encode_plus(' '+content_true, max_length=32, truncation=True, return_tensors='pt',padding='max_length')
            content_prev = self.tokenizer.encode_plus(content_prev, max_length=32, truncation=True, return_tensors='pt', add_special_tokens = self.add_special_tokens, padding='max_length')
        return {
                'content_prev': content_prev['input_ids'][0], 
                'content_prev_mask': content_prev['attention_mask'][0], 
                'additional_bs':torch.tensor(additional_bs, dtype=torch.float32),
                'content_prev_sep':self.tokenizer.encode_plus(['<brain/>', '</brain>'],return_tensors='pt')['input_ids'][0], 
                'content_true': content_true['input_ids'][0], 
                'content_true_mask': content_true['attention_mask'][0], 
                'trail_id': trail_id,
                'content_all': content_all['input_ids'][0],
                'content_all_mask': content_all['attention_mask'][0],
                'id':id
            }
    
    def normalized(self, dic_pere):
        all_data = []
        for story in dic_pere.keys():
            all_data.append(np.array(dic_pere[story]['fmri']))
        all_data = np.concatenate(all_data, axis=0) 
        all_data = self.scaler.fit_transform(all_data)
        start_idx = 0
        for story in dic_pere.keys():
            dic_pere[story]['fmri'] = all_data[start_idx:start_idx+dic_pere[story]['fmri'].shape[0]]
            start_idx += dic_pere[story]['fmri'].shape[0] 
    
    def __init__(self, input_dataset, args, tokenizer,decoding_model=None):
        self.decoding_model = decoding_model
        self.args = args
        self.add_special_tokens=False
        self.inputs = []
        self.shuffle_times = args['shuffle_times']
        dataset_path = args['dataset_path']
        if args['normalized']:
            self.scaler = StandardScaler()
        self.tokenizer = tokenizer
        id2info = {}
        tmp_id = 0
        if 'Pereira' in args['task_name']:
            dataset_name, subject_name = args['task_name'].split('_')
            pere_dataset = pickle.load(open(f'{dataset_path}{subject_name}.pca1000.wq.pkl.dic','rb')) if args['fmri_pca'] else pickle.load(open(f'{dataset_path}{subject_name}.wq.pkl.dic','rb'))
            if args['normalized']:
                self.normalized(pere_dataset)
            for story in input_dataset.keys():
                for item_id, item in enumerate(input_dataset[story]):
                    for k in range(1, len(item['word'])):
                        content_prev = ' '.join([item['word'][j]['content'] for j in range(0,k)])
                        additional_bs = np.array([pere_dataset[story]['fmri'][idx] for idx in item['word'][k]['additional']])
                        content_true = item['word'][k]['content']
                        if args['add_end']:
                            content_true += '<|endoftext|>'
                        random_number = random.random()
                        id2info[tmp_id] = {'story':story, 'item_id':item_id, 'k': k}
                        packed_info = self.pack_info(content_prev, additional_bs, content_true, random_number, id = tmp_id)
                        tmp_id += 1
                        if torch.sum(packed_info['content_true_mask']) > 0:
                            self.inputs.append(packed_info)   
        elif 'Narratives' in args['task_name']:
            subject_name = args['task_name'].split('_')[1]
            content_true2idx = {}
            for story in args['Narratives_stories']:
                Narratives_dataset = pickle.load(open(f'{dataset_path}{story}.pca1000.wq.pkl.dic','rb')) if args['fmri_pca'] else pickle.load(open(f'{dataset_path}{story}.wq.pkl.dic','rb'))
                for subject in [f'sub-{subject_name}']:
                    for item_id, item in enumerate(input_dataset[story][subject]):
                        sid = int(subject.split('-')[1])
                        for k in range(1, len(item['word'])):
                            content_prev = ' '.join([item['word'][j]['content'] for j in range(0,k)])
                            additional_bs = np.array([Narratives_dataset[subject]['fmri'][idx] for idx in item['word'][k]['additional']])
                            content_true = item['word'][k]['content']
                            if len(content_true.strip()) == 0:
                                continue
                            if args['add_end']:
                                content_true += '<|endoftext|>'
                            random_number = random.random()
                            if content_true not in content_true2idx.keys():
                                content_true2idx[content_true] = random_number
                            id2info[tmp_id] = {'story':story, 'item_id':item_id, 'k': k}
                            packed_info = self.pack_info(content_prev, additional_bs, content_true, content_true2idx[content_true], id = tmp_id)
                            tmp_id += 1
                            if len(packed_info['content_true']) > 0 and len(packed_info['content_prev']) > 0:
                                self.inputs.append(packed_info)
        elif 'Huth' in args['task_name']:
            dataset_name = args['task_name'].split('_')[0]
            subject_name = args['task_name'].split('_')[1]
            data_info2 = json.load(open('../dataset_info/Huth.json'))
            data_info2random_number = [0.2 * i for i in range(len(data_info2))]
            ds_dataset = pickle.load(open(f'{dataset_path}{subject_name}.pca1000.wq.pkl.dic','rb'))
            content_true2idx = {}
            for sid, story in enumerate(input_dataset.keys()):
                if story not in data_info2:
                    continue
                for item_id, item in enumerate(input_dataset[story]):
                    for k in range(1, len(item['word'])):
                        content_prev = ' '.join([item['word'][j]['content'] for j in range(0,k)])
                        additional_bs = np.array([ds_dataset[story]['fmri'][idx] for idx in item['word'][k]['additional']])
                        content_true = item['word'][k]['content']
                        if len(content_true.strip()) == 0:
                            continue
                        if content_true not in content_true2idx.keys():
                            content_true2idx[content_true] = random.random()
                        if args['add_end']:
                            content_true += '<|endoftext|>'
                        if args['data_spliting'] == 'cross_story':
                            trail_id = data_info2random_number[data_info2.index(story)]
                        else:
                            trail_id = content_true2idx[content_true]
                        id2info[tmp_id] = {'story':story, 'item_id':item_id, 'k': k}
                        packed_info = self.pack_info(content_prev, additional_bs, content_true, trail_id, id = tmp_id)
                        tmp_id += 1
                        if torch.sum(packed_info['content_true_mask']) > 0:
                            self.inputs.append(packed_info)
                        if self.inputs[-1]['content_prev'].shape[0] == 0 and args['context']:
                            self.inputs = self.inputs[:-1]
        self.pack_data_from_input(args)
        json.dump(id2info, open(self.args['checkpoint_path']+'/'+'id2info.json', 'w'))
    
    def pack_data_from_input(self, args, ):
        self.train = []
        self.test = []
        self.valid = []
        self.is_shuffled = False
        test_ids = args['test_trail_ids']
        valid_ids = args['valid_trail_ids']
        if args['input_method'] == 'permutated':
            tmp_additional_bs = copy.deepcopy([self.inputs[(idx+int(len(self.inputs)/2))%len(self.inputs)]['additional_bs'] for idx in range(len(self.inputs))])
            random.shuffle(tmp_additional_bs)
        for idx,item in enumerate(self.inputs):
            if args['input_method'] == 'permutated':
                item['additional_bs'] = tmp_additional_bs[idx]
            if item['trail_id'] > test_ids[0] and item['trail_id'] <= test_ids[1]:
                self.test.append(item)
            elif item['trail_id'] > valid_ids[0] and item['trail_id'] <= valid_ids[1]:
                self.valid.append(item)
            else:
                self.train.append(item)
        if args['data_size'] != -1:
            random.shuffle(self.train)
            self.train = self.train[:args['data_size']]
        
        self.train_dataset = Splited_FMRI_dataset(self.train, args = args)
        self.valid_dataset = Splited_FMRI_dataset(self.valid, args = args)
        self.test_dataset = Splited_FMRI_dataset(self.test, args = args)
