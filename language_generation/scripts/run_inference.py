import os
# 0827
num_gpus = [0,0,0,1,1]
mask_ratio = 0.5
for idx,u in enumerate(['P01', 'M04', 'M07', 'M15', 'M02']): #, 
    checkpoint_path = f'Pereira_{u}_gpt2-xl_lr1e-5_tid1,2_a0.1f0.25'
    base_operation2 = f'python main.py -task_name Pereira_{u} -cuda {num_gpus[idx]} -load_check_point True -model_name gpt2-xl -mode inference -checkpoint_path {checkpoint_path} -lr 1e-5 -test_trail_ids 1,2 -noise_ratio {mask_ratio} >../../log/0903/Pereira_{u}_tid1,2_inference_ratio{mask_ratio}.txt 2>&1 &'
    os.system(base_operation2)

num_gpus = [1,]
for idx,u in enumerate(['P01',]):
    checkpoint_path = f'Pereira_{u}_gpt2-xl_lr1e-5_tid1,2_a0.1f0.25_random'
    base_operation2 = f'python -u main.py -task_name Pereira_{u} -cuda {num_gpus[idx]} -load_check_point True -mode inference -model_name gpt2-xl -checkpoint_path {checkpoint_path} -lr 1e-5 -test_trail_ids 1,2 -random_input True -noise_ratio {mask_ratio} >../../log/0903/Pereira_{u}_random_tid1,2_inference_ratio{mask_ratio}.txt 2>&1 &'
    os.system(base_operation2)
    