import os

# 0827
# num_gpus = [0,0,1,1,2]
# for idx,u in enumerate(['P01', 'M04', 'M07', 'M15', 'M02']):
#     checkpoint_path = f'Pereira_{u}_gpt2-xl_lr1e-5_tid1,2'
#     base_operation2 = f'python main.py -task_name Pereira_{u} -cuda {num_gpus[idx]} -load_check_point False -model_name gpt2-xl -checkpoint_path {checkpoint_path} -lr 1e-5 -test_trail_ids 1,2 >../../log/faster/0827/Pereira_{u}_tid1,2.txt 2>&1 &'
#     os.system(base_operation2)

# num_gpus = [3,6,6,7,7]
# for idx,u in enumerate(['P01', 'M04', 'M07', 'M15', 'M02']):
#     checkpoint_path = f'Pereira_{u}_gpt2-xl_lr1e-5_tid1,2_random'
#     base_operation2 = f'python -u main.py -task_name Pereira_{u} -cuda {num_gpus[idx]} -load_check_point False -model_name gpt2-xl -checkpoint_path {checkpoint_path} -lr 1e-5 -test_trail_ids 1,2 -random_input True >../../log/faster/0827/Pereira_{u}_random_tid1,2.txt 2>&1 &'
#     os.system(base_operation2)

# 0901
# num_gpus = [0,0,1,1,2,2,3,3]
# for idx,u in enumerate(['F','H','I','J','K','L','M','N']):
#     checkpoint_path = f'HP_{u}_gpt2-xl_lr1e-5_random'
#     base_operation2 = f'python -u main.py -task_name HP_{u} -cuda {num_gpus[idx]} -load_check_point False -model_name gpt2-xl -checkpoint_path {checkpoint_path} -lr 1e-5  -random_input True >../../log/0901/HP_{u}_random.txt 2>&1 &'
#     os.system(base_operation2)

# 0902
# num_gpus = [1,]
# for idx,u in enumerate(['M04']):
#     checkpoint_path = f'Pereira_{u}_gpt2-xl_lr1e-5_tid1,2_a0.1f0.25'
#     base_operation2 = f'python main.py -task_name Pereira_{u} -cuda {num_gpus[idx]} -load_check_point True -model_name gpt2-xl -checkpoint_path {checkpoint_path} -lr 1e-5 -test_trail_ids 1,2 -mode evaluate -fake_input 0.25 -additional_loss 0.1 -project_name a0.1f0.25 >../../log/0902/Pereira_{u}_tid1,2.txt 2>&1 &'
#     os.system(base_operation2)

# num_gpus = [6,]
# for idx,u in enumerate(['P01',]):
#     checkpoint_path = f'Pereira_{u}_gpt2-xl_lr1e-5_tid1,2_a0.1f0.25_random'
#     base_operation2 = f'python -u main.py -task_name Pereira_{u} -cuda {num_gpus[idx]} -load_check_point False -model_name gpt2-xl -checkpoint_path {checkpoint_path} -lr 1e-5 -test_trail_ids 1,2 -fake_input 0.25 -additional_loss 0.1 -random_input True -project_name a0.1f0.25 >../../log/0902/Pereira_{u}_random_tid1,2.txt 2>&1 &'
#     os.system(base_operation2)

# 0901
num_gpus = [1,1,2,2]
for idx,u in enumerate(['L','M','N','J']): # ,
    checkpoint_path = f'HP_{u}_gpt2-xl_lr1e-4'
    base_operation2 = f'python -u main.py -task_name HP_{u} -cuda {num_gpus[idx]} -load_check_point False -model_name gpt2-xl -checkpoint_path {checkpoint_path} -project_name hp4 -lr 1e-4 >../../log/0901/HP_{u}_1e-4.txt 2>&1 &'
    os.system(base_operation2)
    
num_gpus = [7,]
for idx,u in enumerate(['F',]): # ,
    checkpoint_path = f'HP_{u}_gpt2-xl_lr1e-4_random'
    base_operation2 = f'python -u main.py -task_name HP_{u} -cuda {num_gpus[idx]} -load_check_point False -model_name gpt2-xl -checkpoint_path {checkpoint_path} -project_name hp4  -random_input True -lr 1e-4 >../../log/0901/HP_{u}_random_1e-4.txt 2>&1 &'
    os.system(base_operation2)
    