from sklearn.linear_model import Ridge, LinearRegression
import joblib
import sys
import random
import numpy as np
import torch
import pickle
import os
import json
import copy
from sklearn import svm
sys.path.append('../../language_generation/')
from src.model import Decoding_model 
from src.config import get_config
from src.data import FMRI_dataset
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def dataset2xy(dataset):
    X = [input_sample['additional_bs'].numpy().flatten() for input_sample in dataset.inputs]
    y = [input_sample['content_true_mask'].numpy().sum() for input_sample in dataset.inputs]
    return X, y
    
if __name__ == '__main__':
    args = get_config()
    print(args)
    save_name = '../results/'
    for key in args.keys():
        if key not in ['cuda']:
            save_name += key+'('+str(args[key])+')_'
    save_name = save_name[:-1]
    dataset_class = FMRI_dataset
    dataset_name = args['task_name'].split('_')[0]
    subject_name = args['task_name'].split('_')[1]
    if 'example' not in args['task_name']:
        args['dataset_path'] = os.path.join(args['dataset_path'], dataset_name)
    dataset_path = args['dataset_path']

    if 'Huth' in args['task_name']:
        input_dataset = pickle.load(open(f'{dataset_path}/{subject_name}.wq.pkl','rb'))
        decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Pereira' in args['task_name']:
        input_dataset = pickle.load(open(f'{dataset_path}/{subject_name}.wq.pkl','rb'))
        decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Narratives' in args['task_name']:
        u2s = json.load(open(f'../../dataset_info/u2s.json'))
        args['Narratives_stories'] = u2s[f'sub-{subject_name}']
        input_dataset = {}
        for story_name in args['Narratives_stories']:
            input_dataset[story_name] = pickle.load(open(f'{dataset_path}/{story_name}.wq.pkl','rb'))
        decoding_model = Decoding_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)

    print('dataset initialized')
    os.makedirs(f'{args["checkpoint_path"]}', exist_ok=True)

    # 创建岭回归模型实例，alpha是正则化强度
    model_ridge = Ridge(alpha = 1.0)
    # model_ridge = LinearRegression()
    # model_ridge = svm.SVR()
    
    X_train, y_train = dataset2xy(dataset.train_dataset) 
    X_test, y_test = dataset2xy(dataset.test_dataset)

    # # 拟合模型
    model_ridge.fit(X_train, y_train)
    
    # # 保存模型
    mode_path = f'{args["checkpoint_path"]}/model.pkl'
    joblib.dump(model_ridge, mode_path)
    
    model_ridge = joblib.load(mode_path)
    
    # 使用模型进行预测
    y_predict = model_ridge.predict(X_test)
    
    # 计算pair-wise accuracy
    scores = []
    for i in range(5):
        y_predict_random = copy.deepcopy(y_predict)
        random.shuffle(y_predict_random)
        for j in range(len(y_predict_random)):
            diff = np.abs(y_predict_random[j] - y_test[j]) - np.abs(y_predict[j] - y_test[j])
            if diff > 0:
                scores.append(1)
            elif diff == 0:
                scores.append(0.5)
            else:
                scores.append(0)
    
    result_f = open(f'{args["checkpoint_path"]}/result.txt','w')
    print('mean scores:', np.mean(scores), file=result_f)
    print('mean word rate:', np.mean(y_predict), file=result_f)
