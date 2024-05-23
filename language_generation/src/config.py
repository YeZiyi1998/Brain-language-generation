import argparse
import datetime
import json
import os
import wandb

dataset2args = {
    'Huth':{
        'test_trail_ids': [0,0.2],
        'valid_trail_ids': [0,0.2]
    },
    'Pereira':{
        'test_trail_ids': [0,0.2],
        'valid_trail_ids': [0,0.2]
    },
    'Narratives':{
        'test_trail_ids': [0,0.2],
        'valid_trail_ids': [0,0.2]
    }
}

def get_config():
    parser = argparse.ArgumentParser(description='Specify config args for training models for brain language generation')
    parser.add_argument('-model_name', help='choose from {gpt2,gpt2-medium,gpt2-large,gpt2-xl,llama-2, huth}', default = "gpt2" ,required=False)
    parser.add_argument('-method', help='choose from {decoding}', default = "decoding" ,required=False)
    parser.add_argument('-task_name', help='examples: Pereira_example', default = "Pereira_example" ,required=False)
    parser.add_argument('-test_trail_ids', default = "", required=False)
    parser.add_argument('-valid_trail_ids', default = "", required=False)
    parser.add_argument('-random_number', default = 1, type = int, required=False)
    parser.add_argument('-batch_size', default = 1, type = int, required=False)
    parser.add_argument('-fmri_pca', default = "True" ,required=False)
    parser.add_argument('-cuda', default = "0" ,required=False)
    parser.add_argument('-layer', type = int, default = -1 ,required=False)
    parser.add_argument('-num_epochs',  type = int, default = 100 ,required=False)
    parser.add_argument('-lr',  type = float, default = 1e-3, required=False)
    parser.add_argument('-dropout',  type = float, default = 0.5, required=False)
    parser.add_argument('-brain_embed_size',  type = float, default = 1000, required=False)
    parser.add_argument('-checkpoint_path', default = "" ,required=False)
    parser.add_argument('-load_check_point', default = "False" ,required=False)
    parser.add_argument('-enable_grad', default = "False" ,required=False)
    parser.add_argument('-mode', default = "train" , choices = ['train','evaluate', 'all', 'end2end', 'acc', 'only_train'],required=False)
    parser.add_argument('-end2end_part', default = '-1,2', required=False)
    parser.add_argument('-additional_loss', default = 0, type = float, required=False)
    parser.add_argument('-fake_input', default = 0, type = float, required=False)
    parser.add_argument('-add_end', default = "False", required=False)
    parser.add_argument('-context', default = "True", required=False)
    parser.add_argument('-roi_selected', default = "[]", required=False)
    parser.add_argument('-project_name', default = "decoding", required=False)
    parser.add_argument('-noise_ratio', default = 0.5, type=float, required=False)
    parser.add_argument('-wandb', default = 'online', type=str, required=False)
    parser.add_argument('-generation_method', default = 'beam', type=str, required=False)
    parser.add_argument('-pos', default = 'False', type=str, required=False)
    parser.add_argument('-output', default = 'test', type=str, required=False)
    parser.add_argument('-data_spliting', default = 'random', choices = ['random','cross_story','end2end'], type=str, required=False)
    parser.add_argument('-loss', default = 'continuation', choices = ['continuation','all'], type=str, required=False)
    parser.add_argument('-brain_model', default = 'mlp', type=str, required=False)
    parser.add_argument('-weight_decay', default = 1.0, type=float, required=False)
    parser.add_argument('-l2', default = 0.0, type=float, required=False)
    parser.add_argument('-num_layers', default = 2, type=int, required=False)
    parser.add_argument('-evaluate_log', default = 'False', required=False)
    parser.add_argument('-normalized', default = 'False', required=False)
    parser.add_argument('-input_method', default = "normal" , choices = ["normal", "permutated", 'without_brain', 'without_text'],required=False)
    parser.add_argument('-activation', default = "relu" , choices = ["relu",'sigmoid','tanh','relu6'],required=False)
    parser.add_argument('-pretrain_epochs', default = 0, type=int,required=False)
    parser.add_argument('-pretrain_lr', default = 0.001, type=float,required=False)
    parser.add_argument('-data_size', default = -1, type=int,required=False)
    parser.add_argument('-results_path', default = 'results', type=str,required=False)
    parser.add_argument('-dataset_path', default = '../../dataset/', type=str,required=False)
    parser.add_argument('-shuffle_times', default = -1, type=int,required=False)
    parser.add_argument('-prev_mask_len', default = 32, type=int,required=False)
    parser.add_argument('-max_generate_len', default = 32, type=int,required=False)
    parser.add_argument('-early_stop', default = 10, type=int,required=False)
    parser.add_argument('-use_bad_words_ids', default = 'False', type=str,required=False)
    parser.add_argument('-repetition_penalty', default = 2.0, type=float, required=False)    
    parser.add_argument('-ncontext', default = 10, type=int, required=False) 
    parser.add_argument('-gcontext', default = 30, type=int, required=False) 
    parser.add_argument('-use_decoder_vocab', default = 'True', type=str, required=False) 
    parser.add_argument('-num_steps', default = 1000, type=int, required=False) 
    parser.add_argument('-length_penalty', default = 0.3, type=float, required=False) 
    parser.add_argument('-beam_width', default = 5, type=int, required=False) 
    parser.add_argument('-extensions', default = 5, type=int, required=False) 
    args = vars(parser.parse_args())
    args['fmri_pca'] = args['fmri_pca'] == 'True'
    args['load_check_point'] = args['load_check_point'] == 'True'
    args['enable_grad'] = args['enable_grad'] == 'True'
    args['add_end'] = args['add_end'] == 'True'
    args['context'] = args['context'] == 'True'
    args['pos'] = args['pos'] == 'True'
    args['normalized'] = args['normalized'] == 'True'
    args['evaluate_log'] = args['evaluate_log'] == 'True'
    args['use_bad_words_ids'] = args['use_bad_words_ids'] == 'True'
    args['use_decoder_vocab'] = args['use_decoder_vocab'] == 'True'
    args['roi_selected'] = json.loads(args['roi_selected'])
    args['end2end_part'] = [float(item) for item in args['end2end_part'].split(',')]
    tmp_dataset2args = dataset2args[args['task_name'].split('_')[0]]
    for k, v in tmp_dataset2args.items():
        if k not in args.keys() or args[k] == '':
            args[k] = v
    args['test_trail_ids'] = [args['test_trail_ids'][0]+args['random_number']*0.2, args['test_trail_ids'][1]+args['random_number']*0.2]
    args['valid_trail_ids'] = [args['valid_trail_ids'][0]+args['random_number']*0.2, args['valid_trail_ids'][1]+args['random_number']*0.2]
    
    # set shuffle_times
    shuffle_times = 10 if 'Huth' not in args['task_name'] and 'Narratives' not in args['task_name'] else 1
    args['shuffle_times'] = shuffle_times if args['shuffle_times'] == 1 else shuffle_times
    
    # manage checkpoint_path
    results_path =  args['results_path']
    if args['checkpoint_path'] == '':
        args['checkpoint_path'] = f'../{results_path}/tmp'
    elif f'../{results_path}/' not in args['checkpoint_path']:
        args['checkpoint_path'] = f'../{results_path}/' + args['checkpoint_path']
    if os.path.exists(args['checkpoint_path']) == False:
        os.makedirs(args['checkpoint_path'])
    args["llm_model_path"] = args['checkpoint_path']
    
    print(args['checkpoint_path'])
    
    # write info
    if args['mode'] in ['train', 'all', 'end2end']:
        json.dump(args, open(args['checkpoint_path']+'/info.json', 'w'))
    
    # setting wandb environment
    os.environ['WANDB_MODE'] = args['wandb']
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_API_PORT'] = '12358'
    if args['wandb'] != 'none':
        wandb.init(
            project=args['project_name'],
            config=args
        )
    return args

