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
    
    def pack_info(self, content_prev, additional_bs, content_true, trail_id):
        if self.args['model_name'] in ['llama-7b','llama-7b-old']:
            # bug 1008
            self.add_special_tokens = True
            content_all = self.tokenizer.encode_plus(content_prev+' '+content_true, max_length=64, truncation=True, return_tensors='pt', add_special_tokens = self.add_special_tokens, padding='max_length',)
            content_all['input_ids'] = content_all['input_ids'][:,1:]
            content_all['attention_mask'] = content_all['attention_mask'][:,1:]
            content_true = self.tokenizer.encode_plus(content_true.strip(), max_length=32, truncation=True, return_tensors='pt',padding='max_length',)
            content_true['input_ids'] = content_true['input_ids'][:,1:]
            content_true['attention_mask'] = content_true['attention_mask'][:,1:]
            content_prev = self.tokenizer.encode_plus(content_prev, max_length=32, truncation=True, return_tensors='pt', add_special_tokens = self.add_special_tokens, padding='max_length', )
            # bug 1001
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
        if args['normalized']:
            self.scaler = StandardScaler()
        self.tokenizer = tokenizer
        if 'HP' in args['task_name']:
            content_true2idx = {}
            # idx = 0
            for item_id, item in enumerate(input_dataset):
                for k in range(1, len(item['word'])):
                    content_prev = ' '.join([item['word'][j]['content'] for j in range(0,k)])
                    additional_bs = item['word'][k]['additional']
                    content_true = item['word'][k]['content']
                    if content_true not in content_true2idx.keys():
                        if args['hp_spliting'] == 'normal':
                            content_true2idx[content_true] = 1 if random.random() < 0.2 else 0
                        elif args['hp_spliting'] == 'random':
                            content_true2idx[content_true] = 1 if random.random() > 0.2 and random.random() < 0.4 else 0
                        elif args['hp_spliting'] == 'last':
                            content_true2idx[content_true] = 1 if item['trail_id'] == 3 else 0
                    idx = content_true2idx[content_true]
                    if args['add_end']:
                        content_true += '<|endoftext|>'
                    self.inputs.append(self.pack_info(content_prev, additional_bs, content_true, idx))
                # idx += 1
        elif 'Pereira' in args['task_name']:
            # use data from all subjects 
            dataset_name, subject_name = args['task_name'].split('_')
            pere_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name}.pca1000.wq.pkl.dic','rb')) if args['fmri_pca'] else pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name}.wq.pkl.dic','rb'))
            if args['normalized']:
                self.normalized(pere_dataset)
            if args['ds_spliting'] != 'cross_subject':
                for story in input_dataset.keys():
                    for item in input_dataset[story]:
                        for k in range(1, len(item['word'])):
                            content_prev = ' '.join([item['word'][j]['content'] for j in range(0,k)])
                            additional_bs = np.array([pere_dataset[story]['fmri'][idx] for idx in item['word'][k]['additional']])
                            content_true = item['word'][k]['content']
                            if args['add_end']:
                                content_true += '<|endoftext|>'
                            # 如果是ds，需要去下载一下
                            # packed_info = self.pack_info(content_prev, additional_bs, content_true, item['trail_id']) # 
                            random_number = random.random()
                            packed_info = self.pack_info(content_prev, additional_bs, content_true, 1 if random_number > 0.2 * args['random_number'] and random_number < 0.2 + 0.2 * args['random_number'] else 0)
                            if torch.sum(packed_info['content_true_mask']) > 0:
                                self.inputs.append(packed_info)
            else:
                content_true2idx = {}
                for candidate_subject_name in ['P01','M04', 'M07', 'M15', 'M02']:
                    trail_id = subject_name == candidate_subject_name
                    input_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{candidate_subject_name}.wq.pkl','rb'))
                    for story in input_dataset.keys():
                        for item in input_dataset[story]:
                            for k in range(1, len(item['word'])):
                                content_prev = ' '.join([item['word'][j]['content'] for j in range(0,k)])
                                additional_bs = np.array([pere_dataset[story]['fmri'][idx] for idx in item['word'][k]['additional']])
                                content_true = item['word'][k]['content']
                                random_number = random.random()
                                if content_true not in content_true2idx.keys():
                                    content_true2idx[content_true] = 1 if random_number > 0.2 and random_number < 0.4 else 0
                                if content_true2idx[content_true] != trail_id:
                                    continue
                                if args['add_end']:
                                    content_true += '<|endoftext|>'
                                packed_info = self.pack_info(content_prev, additional_bs, content_true, trail_id)
                                if torch.sum(packed_info['content_true_mask']) > 0:
                                    self.inputs.append(packed_info)     
        elif 'Narratives_person' in args['task_name']:
            subject_name = args['task_name'].split('_')[2]
            content_true2idx = {}
            for story in args['Narratives_stories']:
                Narratives_dataset = pickle.load(open(f'../../dataset/preprocessed/Narratives/{story}.pca1000.wq.pkl.dic','rb')) if args['fmri_pca'] else pickle.load(open(f'../../dataset/preprocessed/Narratives/{story}.wq.pkl.dic','rb'))
                # cross-subject and cross-story
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
                                content_true2idx[content_true] = 1 if random_number > 0.2 and random_number < 0.4 else 0
                            
                            if args['ds_spliting'] == 'cross_story':
                                packed_info = self.pack_info(content_prev, additional_bs, content_true, 1 if story == args['Narratives_test_story'] else 0) # 1 item['trail_id']
                            else:
                                # 1001
                                packed_info = self.pack_info(content_prev, additional_bs, content_true, content_true2idx[content_true])
                                # packed_info = (self.pack_info(content_prev, additional_bs, content_true, 1 if item_id > len(input_dataset[story][subject]) / 5 * 4 else 0)) 
                            if len(packed_info['content_true']) > 0 and len(packed_info['content_prev']) > 0:
                                self.inputs.append(packed_info)
            random.shuffle(self.inputs)
            self.inputs = self.inputs[:15000]
        elif 'Narratives_cross' in args['task_name']:
            subject_name = args['task_name'].split('_')[2]
            content_true2idx = {}
            for story in args['Narratives_stories']:
                Narratives_dataset = pickle.load(open(f'../../dataset/preprocessed/Narratives/{story}.pca1000.wq.pkl.dic','rb')) if args['fmri_pca'] else pickle.load(open(f'../../dataset/preprocessed/Narratives/{story}.wq.pkl.dic','rb'))
                # cross-subject and cross-story
                for subject in input_dataset[story].keys():
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
                                content_true2idx[content_true] = 1 if random_number > 0.2 and random_number < 0.4 else 0
                            trail_id = content_true2idx[content_true]
                            if trail_id == 0 and subject == f'sub-{subject_name}' or trail_id == 1 and subject != f'sub-{subject_name}':
                                continue
                            if args['ds_spliting'] == 'cross_story':
                                packed_info = self.pack_info(content_prev, additional_bs, content_true, 1 if story == args['Narratives_test_story'] else 0) # 1 item['trail_id']
                            else:
                                packed_info = self.pack_info(content_prev, additional_bs, content_true, trail_id)
                            if len(packed_info['content_true']) > 0 and len(packed_info['content_prev']) > 0:
                                self.inputs.append(packed_info)
            random.shuffle(self.inputs)
            self.inputs = self.inputs[:15000]
        elif 'Narratives' in args['task_name']:
            exclude_subjects = input_dataset[args['Narratives_test_story']].keys()
            content_true2idx = {}
            for story in args['Narratives_stories']: # ['pieman', "tunnel", "lucy", "prettymouth", "notthefallintact", "merlin", '21styear', 'sherlock', "slumlord", "reach"]
                Narratives_dataset = pickle.load(open(f'../../dataset/preprocessed/Narratives/{story}.pca1000.wq.pkl.dic','rb')) if args['fmri_pca'] else pickle.load(open(f'../../dataset/preprocessed/Narratives/{story}.wq.pkl.dic','rb'))
                # cross-subject and cross-story
                for subject in input_dataset[story].keys():
                    if subject in exclude_subjects and story != args['Narratives_test_story']:
                        continue
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
                                content_true2idx[content_true] = 1 if random_number > 0.2 and random_number < 0.4 else 0
                            
                            if args['ds_spliting'] == 'cross_story':
                                packed_info = self.pack_info(content_prev, additional_bs, content_true, 1 if story == args['Narratives_test_story'] else 0) # 1 item['trail_id']
                            else:
                                # 1001
                                packed_info = self.pack_info(content_prev, additional_bs, content_true, content_true2idx[content_true])
                                # packed_info = (self.pack_info(content_prev, additional_bs, content_true, 1 if item_id > len(input_dataset[story][subject]) / 5 * 4 else 0)) 
                            if len(packed_info['content_true']) > 0 and len(packed_info['content_prev']) > 0:
                                self.inputs.append(packed_info)
            random.shuffle(self.inputs)
            self.inputs = self.inputs[:15000]

        elif 'ds003020' in args['task_name']:
            dataset_name = args['task_name'].split('_')[0]
            subject_name = args['task_name'].split('_')[1]
            # data_info = json.load(open('../dataset_info/ds003020.json'))
            data_info2 = json.load(open('../dataset_info/ds003020_new.json'))
            if args['ds_spliting'] != 'cross_subject':  
                ds_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name}.pca1000.wq.pkl.dic','rb'))
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
                                content_true2idx[content_true] = 1 if random.random() < 0.2 else 0
                            if args['add_end']:
                                content_true += '<|endoftext|>'
                            # 如果是ds，需要去下载一下
                            if args['ds_spliting'] == 'cross_story':
                                trail_id = 1 if story in data_info2[-5:] else 0
                            else:
                                trail_id = content_true2idx[content_true]
                            packed_info = self.pack_info(content_prev, additional_bs, content_true, trail_id)
                            if torch.sum(packed_info['content_true_mask']) > 0:
                                self.inputs.append(packed_info)
                            if self.inputs[-1]['content_prev'].shape[0] == 0 and args['context']:
                                self.inputs = self.inputs[:-1]
            else:
                for subject_name_all in ['2','1','4','5','6','7','8','3',]: # 
                    ds_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name_all}.pca1000.wq.pkl.dic','rb'))
                    content_true2idx = {}
                    for story in input_dataset.keys():
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
                                    content_true2idx[content_true] = 1 if random.random() < 0.2 else 0
                                if args['add_end']:
                                    content_true += '<|endoftext|>'
                                # 如果是ds，需要去下载一下
                                if args['ds_spliting'] == 'cross_story':
                                    trail_id = 1 if story in data_info2[-5:] else 0
                                else:
                                    trail_id = content_true2idx[content_true]
                                if trail_id == 0 and subject_name_all == subject_name or trail_id == 1 and subject_name_all != subject_name:
                                    continue
                                packed_info = self.pack_info(content_prev, additional_bs, content_true, trail_id)
                                if torch.sum(packed_info['content_true_mask']) > 0:
                                    self.inputs.append(packed_info)
                                if self.inputs[-1]['content_prev'].shape[0] == 0 and args['context']:
                                    self.inputs = self.inputs[:-1]
        self.pack_data_from_input(args)
        if args['use_bad_words_ids']:
            self.get_bad_word_ids()
    
    def get_bad_word_ids(self,):
        vocabulary = np.unique([item['content_true'] for item in self.test])
        print('length of vocabulary: ', len(vocabulary))
        # np.array(self.tokenizer.get_vocab().keys())
        self.bad_word_ids = np.setdiff1d(np.array(list(self.tokenizer.get_vocab().values())), vocabulary)
    
    def pack_data_from_input(self, args, ):
        self.train = []
        self.test = []
        self.valid = []
        self.is_shuffled = False
        if args['data_splitting'] == 'trail_id':
            test_ids = args['test_trail_ids']
            valid_ids = args['valid_trail_ids']
            if args['input_method'] == 'random_all_input':
                random_all_input = torch.rand([len(self.inputs)]+list(self.inputs[0]['additional_bs'].shape), dtype=torch.float32) * 100
            if args['input_method'] == 'shuffle_input':
                tmp_additional_bs = copy.deepcopy([self.inputs[(idx+int(len(self.inputs)/2))%len(self.inputs)]['additional_bs'] for idx in range(len(self.inputs))])
                random.shuffle(tmp_additional_bs)
            for idx,item in enumerate(self.inputs):
                if args['input_method'] == 'shuffle_input':
                    item['additional_bs'] = tmp_additional_bs[idx]
                elif args['input_method'] == 'random_input':
                    item['additional_bs'] = torch.rand(item['additional_bs'].shape, dtype=torch.float32)
                elif args['input_method'] == 'random_all_input':
                    item['additional_bs'] = random_all_input[idx]
                # ! HP数据集和huth数据集也只用trail id为1的吗？
                if item['trail_id'] % 10 in test_ids:
                    self.test.append(item)
                elif item['trail_id'] % 10 in valid_ids:
                    self.valid.append(item)
                else:
                    self.train.append(item)
        if args['data_size'] != -1:
            random.shuffle(self.train)
            self.train = self.train[:args['data_size']]
        
        if args['mode'] == 'inference':
            self.test_dataset = Splited_FMRI_dataset(self.add_context_noise(self.test), args = args)
        elif 'evaluate_length' in args['mode']:
            self.test = self.expand(self.test)
            self.test_dataset = Splited_FMRI_dataset(self.test, args = args)
        else:
            # bug narratives只用2000数据
            if 'Narratives' in args['task_name'] and 'Narratives_person' not in args['task_name']:
                self.train_dataset = Splited_FMRI_dataset(self.train, most_epoch=2000, args = args)
            else:
                self.train_dataset = Splited_FMRI_dataset(self.train, most_epoch=20000, args = args)
            if len(self.valid) > 0:
                self.valid_dataset = Splited_FMRI_dataset(self.valid, most_epoch=8000, args = args)
            else:
                self.valid_dataset = Splited_FMRI_dataset(self.test, most_epoch=8000, args = args)
            self.test_dataset = Splited_FMRI_dataset(self.test, args = args)
        if args['input_method'] == 'shuffle_valid':
            self.shuffle_valid()
    
    def shuffle_valid(self,):
        if self.is_shuffled:
            return
        self.is_shuffled = True
        shuffle_times = 10 if 'ds003020' not in self.args['task_name'] and 'Narratives' not in self.args['task_name'] else 1
        self.test_new = []
        shuffle_times = 1 if 'evaluate_length' in self.args['mode'] or 'encoding' in self.args['method'] else shuffle_times
        tmp_additional_bs = copy.deepcopy([self.test[idx]['additional_bs'] for idx in range(len(self.test))])
        for i in range(shuffle_times):
            for item in self.test:
                self.test_new.append(copy.deepcopy(item))
                self.test_new[-1]['additional_bs'] = tmp_additional_bs[random.randint(0, len(self.test)-1)]
        self.test_dataset_old = Splited_FMRI_dataset(copy.deepcopy(self.test), args = self.args)
        self.test = self.test_new
        self.test_dataset = Splited_FMRI_dataset(self.test, args = self.args)
     
    def expand(self, test_dataset):
        expand_dataset = []
        for item in test_dataset:
            content_true_length = torch.sum(item['content_true_mask'])
            content_pre_length = torch.sum(item['content_prev_mask'])
            for j in range(content_true_length - 1):
                if content_pre_length < len(item['content_prev']):
                    item['content_prev'][content_pre_length] = item['content_true'][0]
                    item['content_prev_mask'][content_pre_length] = 1
                    content_pre_length += 1
                else:
                    item['content_prev'] = torch.cat([item['content_prev'][:-1], item['content_true'][0].unsqueeze(0)])
                item['content_true'] = torch.cat([item['content_true'][1:], item['content_true'][-1:]])
                item['content_true_mask'][content_true_length-1] = 0
                content_true_length -= 1
                expand_dataset.append(copy.deepcopy(item))
        return expand_dataset
