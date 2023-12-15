import argparse
import datetime
import json
import os
import wandb

dataset2args = {
    'HP':{
        'data_splitting': 'trail_id',
        'test_trail_ids': '1',
    },
    'ds003020':{
        'data_splitting': 'trail_id',
        'test_trail_ids': '1,2',
    },
    'Pereira':{
        'data_splitting': 'trail_id',
        'test_trail_ids': '1,2',
    },
    'Narratives':{
        'data_splitting': 'trail_id',
        'test_trail_ids': '1',
    }
}

def get_config():
    parser = argparse.ArgumentParser(description='Specify config args for training models for brain language generation')
    parser.add_argument('-model_name', help='choose from {gpt2,gpt2-medium,gpt2-large,gpt2-xl,llama-2}', default = "gpt2" ,required=False)
    parser.add_argument('-method', help='choose from {decoding}', default = "decoding" ,required=False)
    parser.add_argument('-task_name', help='choose from {HP_F,HP_H,ds003020_1,...,Pereira_M02}', default = "ds003020_1" ,required=False)
    parser.add_argument('-test_trail_ids', default = "" , required=False)
    parser.add_argument('-valid_trail_ids', default = "" , required=False)
    parser.add_argument('-random_number', default = 1, type = int, required=False)
    parser.add_argument('-batch_size', default = 1, type = int, required=False)
    parser.add_argument('-fmri_pca', default = "True" ,required=False)
    parser.add_argument('-cuda', default = "3" ,required=False)
    parser.add_argument('-layer', type = int, default = -1 ,required=False)
    parser.add_argument('-num_epochs',  type = int, default = 100 ,required=False)
    parser.add_argument('-lr',  type = float, default = 1e-3, required=False)
    parser.add_argument('-dropout',  type = float, default = 0.5, required=False)
    parser.add_argument('-checkpoint_path', default = "" ,required=False)
    parser.add_argument('-without_input', default = "False" ,required=False)
    parser.add_argument('-load_check_point', default = "False" ,required=False)
    parser.add_argument('-enable_grad', default = "False" ,required=False)
    parser.add_argument('-mode', default = "train" , choices = ['train', 'valid','evaluate', 'inference','evaluate_length','only_train', 'all','evaluate_all','evaluate_length_valid','evaluate_length_test','distribution'],required=False)
    parser.add_argument('-model_split', default = -1, type = int, required=False)
    parser.add_argument('-additional_loss', default = 0, type = float, required=False)
    parser.add_argument('-fake_input', default = 0, type = float, required=False)
    parser.add_argument('-add_end', default = "False", required=False)
    parser.add_argument('-context', default = "True", required=False)
    parser.add_argument('-roi_selected', default = "[]", required=False)
    parser.add_argument('-project_name', default = "decoding", required=False)
    parser.add_argument('-noise_ratio', default = 0.5, type=float, required=False)
    parser.add_argument('-wandb', default = 'online', type=str, required=False)
    parser.add_argument('-generation_method', default = 'beam', type=str, required=False)
    parser.add_argument('-hp_spliting', default = 'normal', choices=['normal','random','last'], type=str, required=False)
    parser.add_argument('-pos', default = 'False', type=str, required=False)
    parser.add_argument('-output', default = 'test', type=str, required=False)
    parser.add_argument('-ds_spliting', default = 'within_story', choices = ['within_story', 'cross_story','cross_subject'], type=str, required=False)
    parser.add_argument('-brain_model', default = 'mlp', type=str, required=False)
    parser.add_argument('-weight_decay', default = 1.0, type=float, required=False)
    parser.add_argument('-l2', default = 0.0, type=float, required=False)
    parser.add_argument('-num_layers', default = 2, type=int, required=False)
    parser.add_argument('-evaluate_log', default = 'False', required=False)
    parser.add_argument('-normalized', default = 'False', required=False)
    parser.add_argument('-input_method', default = "normal" , choices = ["normal", "shuffle_input", "random_input", "random_all_input", "mask_input", 'shuffle_valid'],required=False)
    parser.add_argument('-activation', default = "relu" , choices = ["relu",'sigmoid','tanh','relu6'],required=False)
    parser.add_argument('-pretrain_epochs', default = 0, type=int,required=False)
    parser.add_argument('-pretrain_lr', default = 0.001, type=float,required=False)
    parser.add_argument('-data_size', default = -1, type=int,required=False)
    parser.add_argument('-without_text', default = "False", type=str,required=False)
    parser.add_argument('-results_path', default = 'results_1018', type=str,required=False)
    parser.add_argument('-story_size', default = -1, type=int,required=False)
    parser.add_argument('-use_bad_words_ids', default = 'False', required=False)
    args = vars(parser.parse_args())
    args['fmri_pca'] = args['fmri_pca'] == 'True'
    args['without_input'] = args['without_input'] == 'True'
    args['load_check_point'] = args['load_check_point'] == 'True'
    args['enable_grad'] = args['enable_grad'] == 'True'
    args['add_end'] = args['add_end'] == 'True'
    args['context'] = args['context'] == 'True'
    args['pos'] = args['pos'] == 'True'
    args['without_text'] = args['without_text'] == 'True'
    args['normalized'] = args['normalized'] == 'True'
    args['evaluate_log'] = args['evaluate_log'] == 'True'
    args['use_bad_words_ids'] = args['use_bad_words_ids'] == 'True'
    args['roi_selected'] = json.loads(args['roi_selected'])
    tmp_dataset2args = dataset2args[args['task_name'].split('_')[0]]
    for k, v in tmp_dataset2args.items():
        if k not in args.keys() or args[k] == '':
            args[k] = v
    args['test_trail_ids'] = list([int(item) for item in args['test_trail_ids'].split(',')])
    args['valid_trail_ids'] = list([int(item) for item in args['valid_trail_ids'].split(',')]) if args['valid_trail_ids'] != '' else []
    results_path =  args['results_path']
    if args['checkpoint_path'] == '':
        args['checkpoint_path'] = f'../{results_path}/tmp'
    elif args['checkpoint_path'] == 'time':
        args['checkpoint_path'] = f'../{results_path}/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    elif f'../{results_path}/' not in args['checkpoint_path']:
        args['checkpoint_path'] = f'../{results_path}/' + args['checkpoint_path']
    if os.path.exists(args['checkpoint_path']) == False:
        os.makedirs(args['checkpoint_path'])
    if args['mode'] in ['train', 'only_train','all']:
        json.dump(args, open(args['checkpoint_path']+'/info.json', 'w'))
    os.environ['WANDB_MODE'] = args['wandb']
    if os.path.exists('/home/yzy')==False and os.path.exists('/home/whs145')==False:
        os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_API_PORT'] = '12358'
    if args['wandb'] != 'none':
        wandb.init(
            # set the wandb project where this run will be logged
            project=args['project_name'],
            # track hyperparameters and run metadata
            config=args
        )
    return args

