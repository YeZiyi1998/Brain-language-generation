from config import get_config
from data import FMRI_dataset
import pickle
import random
import numpy as np
import torch
import json
from model import Decoding_model 
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = get_config()
    print(args)
    save_name = '../results/'
    for key in args.keys():
        if key not in ['cuda']:
            save_name += key+'('+str(args[key])+')_'
    save_name = save_name[:-1]
    dataset_class = FMRI_dataset
    dataset_path = args['dataset_path']

    if 'Huth' in args['task_name']:
        dataset_name = args['task_name'].split('_')[0]
        subject_name = args['task_name'].split('_')[1]
        if args['fmri_pca']:
            input_dataset = pickle.load(open(f'{dataset_path}/{subject_name}.pca1000.wq.pkl','rb'))
            args['brain_embed_size'] = 1000
        else:
            input_dataset = pickle.load(open(f'{dataset_path}/{subject_name}.wq.pkl','rb'))
            args['brain_embed_size'] = input_dataset[0]['word'][0]['additional'].shape
        decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Pereira' in args['task_name']:
        dataset_name = args['task_name'].split('_')[0]
        subject_name = args['task_name'].split('_')[1]
        input_dataset = pickle.load(open(f'{dataset_path}/{subject_name}.wq.pkl','rb'))
        if args['fmri_pca']:
            args['brain_embed_size'] = 1000
        else:
            input_dataset = pickle.load(open(f'{dataset_path}/{subject_name}.wq.pkl','rb'))
            args['brain_embed_size'] = input_dataset[0]['word'][0]['additional'].shape
        decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Narratives' in args['task_name']:
        u2s = json.load(open('{dataset_path}Narratives/u2s.json'))
        args['Narratives_stories'] = u2s[f'sub-{args["task_name"].split("_")[2]}']
        input_dataset = {}
        for story_name in args['Narratives_stories']:
            input_dataset[story_name] = pickle.load(open(f'{dataset_path}/{story_name}.wq.pkl','rb'))
        if args['fmri_pca']:
            args['brain_embed_size'] = 1000
        else:
            args['brain_embed_size'] = input_dataset[0]['word'][0]['additional'].shape
        decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)

    print('dataset initialized')
    
    if args['load_check_point']:
        decoding_model.prompt_model.check_point = decoding_model.check_point
    decoding_model.prompt_model.init_encoding_model()
    
    if args['mode'] in ['train','only_train','all']:
        decoding_model.train(dataset.train_dataset, dataset.valid_dataset)
    if args['mode'] in ['all','evaluate',]:
        args['mode'] = 'evaluate' if args['mode'] == 'train' else args['mode']
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        decoding_model.prompt_model.check_point = decoding_model.check_point
        decoding_model.prompt_model.init_encoding_model()
        decoding_model.test(dataset.test_dataset, args['output'])

