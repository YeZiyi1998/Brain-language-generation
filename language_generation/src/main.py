from config import get_config
from data import FMRI_dataset, ROI_dataset
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
    decoding_model = None
    dataset_class = ROI_dataset if len(args['roi_selected']) > 0 else FMRI_dataset

    if 'ds003020' in args['task_name']:
        dataset_name = args['task_name'].split('_')[0]
        subject_name = args['task_name'].split('_')[1]
        if len(args['roi_selected']) > 0:
            input_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name}.pca1000.wq.pkl','rb'))
            roi_dimension_dic = pickle.load(open('/home/bingxing2/home/scx7140/fmri/dataset/preprocessed/ds003020/roi_f2.dic', 'rb'))[int(subject_name)]
            args['brain_embed_size'] = np.sum([roi_dimension_dic[item] for item in args['roi_selected']])
        elif args['fmri_pca']:
            input_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name}.pca1000.wq.pkl','rb'))
            args['brain_embed_size'] = 1000
        else:
            input_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name}.wq.pkl','rb'))
            args['brain_embed_size'] = input_dataset[0]['word'][0]['additional'].shape
        if decoding_model is None:
            decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'HP' in args['task_name']:
        dataset_name = args['task_name'].split('_')[0]
        subject_name = args['task_name'].split('_')[1]
        if args['fmri_pca']:
            input_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name}.pca1000.wq.pkl','rb'))
            args['brain_embed_size'] = 1000
        else:
            input_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name}.wq.pkl','rb'))
            args['brain_embed_size'] = input_dataset[0]['word'][0]['additional'].shape
        if decoding_model is None:
            decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Pereira' in args['task_name']:
        dataset_name = args['task_name'].split('_')[0]
        subject_name = args['task_name'].split('_')[1]
        input_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name}.wq.pkl','rb'))
        if args['fmri_pca']:
            args['brain_embed_size'] = 1000
        else:
            input_dataset = pickle.load(open(f'../../dataset/preprocessed/{dataset_name}/{subject_name}.wq.pkl','rb'))
            args['brain_embed_size'] = input_dataset[0]['word'][0]['additional'].shape
        if decoding_model is None:
            decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Narratives_person' in args['task_name'] or 'Narratives_cross' in args['task_name']:
        # args['Narratives_stories'] = ['pieman', 'lucy', "prettymouth", "tunnel", ] 
        # args['Narratives_stories'] = ['pieman', "tunnel", "lucy", "prettymouth", "notthefallintact", "merlin", '21styear', 'sherlock', "slumlord", "reach"]
        u2s = json.load(open('../../dataset/preprocessed/Narratives/u2s.json'))
        args['Narratives_stories'] = u2s[f'sub-{args["task_name"].split("_")[2]}']
        input_dataset = {}
        for story_name in args['Narratives_stories']:
            input_dataset[story_name] = pickle.load(open(f'../../dataset/preprocessed/Narratives/{story_name}.wq.pkl','rb'))
        if args['fmri_pca']:
            args['brain_embed_size'] = 1000
        else:
            args['brain_embed_size'] = input_dataset[0]['word'][0]['additional'].shape
        if decoding_model is None:
            decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Narratives' in args['task_name']:
        # args['Narratives_stories'] = ['pieman', 'lucy', "prettymouth", "tunnel", ] 
        # args['Narratives_stories'] = ['pieman', "tunnel", "lucy", "prettymouth", "notthefallintact", "merlin", '21styear', 'sherlock', "slumlord", "reach"]
        args['Narratives_test_story'] = args['task_name'].split('_')[1]
        if args['Narratives_test_story'] == 'all':
            args['Narratives_stories'] = ['pieman', 'lucy', "prettymouth", "tunnel", ]
        else:
            args['Narratives_stories'] = [args['Narratives_test_story']]
        input_dataset = {}
        for story_name in args['Narratives_stories']:
            input_dataset[story_name] = pickle.load(open(f'../../dataset/preprocessed/Narratives/{story_name}.wq.pkl','rb'))
        if args['fmri_pca']:
            args['brain_embed_size'] = 1000
        else:
            args['brain_embed_size'] = input_dataset[0]['word'][0]['additional'].shape
        if decoding_model is None:
            decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Zuco' in args['task_name']:
        input_dataset = pickle.load(open('../../EEG-To-Text/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle' , 'rb'))  
    elif 'UERCM' in args['task_name']:
        input_dataset = pickle.load(open('../../EEG-To-Text/dataset/UERCM/all.pkl', 'rb'))
        pass

    print('dataset initialized')
    
    if args['load_check_point']:
        decoding_model.prompt_model.check_point = decoding_model.check_point
    decoding_model.prompt_model.init_encoding_model()
    
    if args['use_bad_words_ids']:
        decoding_model.prompt_model.bad_words_ids = np.array(dataset.bad_word_ids).reshape(-1,1).tolist()
    
    if args['mode'] in ['train','only_train','all']:
        decoding_model.train(dataset.train_dataset, dataset.test_dataset)
    if args['mode'] in ['train', 'all','evaluate','evaluate_all']:
        args['mode'] = 'evaluate' if args['mode'] == 'train' else args['mode']
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        decoding_model.prompt_model.check_point = decoding_model.check_point
        decoding_model.prompt_model.init_encoding_model()
        if 'encoding' not in args['method']:
            decoding_model.test(dataset.test_dataset, args['output'])
        else:
            # bug 后来才加了.encoding
            decoding_model.test(dataset.test_dataset, dataset.test_dataset_old, args['output'] + '.encoding')
    if args['mode'] == 'inference':
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        decoding_model.prompt_model.check_point = decoding_model.check_point
        decoding_model.prompt_model.init_encoding_model()
        decoding_model.test(dataset.test_dataset, 'inference'+str(args['noise_ratio']))
    if args['mode'] in ['all','evaluate_all'] and 'encoding' not in args['method']:
        dataset.shuffle_valid()
        decoding_model.test(dataset.test_dataset, 'valid')
    elif 'evaluate_length' in args['mode']:
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        decoding_model.prompt_model.check_point = decoding_model.check_point
        decoding_model.prompt_model.init_encoding_model()
        if 'test' in args['mode']:
            output_file = 'test_length'
            if 'encoding' in args['method']:
                output_file += '.encoding'
                decoding_model.test(dataset.test_dataset, dataset.test_dataset_old, output_file)
            else:
                decoding_model.test(dataset.test_dataset, output_file)
        elif 'valid' in args['mode']:
            dataset.shuffle_valid()
            decoding_model.test(dataset.test_dataset, 'valid_length')
        else:
            decoding_model.test(dataset.test_dataset, 'test_length')
            dataset.shuffle_valid()
            decoding_model.test(dataset.test_dataset, 'valid_length')
    elif 'distribution' in args['mode']:
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        decoding_model.prompt_model.check_point = decoding_model.check_point
        decoding_model.prompt_model.init_encoding_model()
        decoding_model.test_distribution(dataset.test_dataset, 'test.distribution')
        dataset.shuffle_valid()
        decoding_model.test_distribution(dataset.test_dataset, 'valid.distribution')
        




