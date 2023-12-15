# import os
# file_names = os.listdir('../results')
# for file_name in file_names:
#     file_list = os.listdir(f'../results/{file_name}')
#     if len(file_list) == 1:
#         os.system(f'mv ../results/{file_name} ../../history/decoding0918/results_bingxing')


# import os
# file_names = os.listdir('../results/pre_09_30')
# for file_name in file_names:
#     # os.system(f'mv ../results/{file_name}/* ../results/pre_09_30/{file_name}')
#     os.system(f'rm -r ../results/{file_name}')


import os
# # file_names = os.listdir('../results/')
# user_list = ['F','H','I','J','K','L','M','N']
# model_name = 'gpt2-xl'
# great_list = ['1001_HP_',f'_{model_name}_lr1e-4_b8_pos_pre10']
# for user in user_list:
#     file_name = great_list[0] + user + great_list[1]
#     # os.system(f'mv ../results/{file_name}/* ../results/pre_09_30/{file_name}')
#     os.system(f'cp -r ../results/{file_name} ../final/_{model_name}')

# user_list = ['P01','M04', 'M07', 'M15', 'M02']
# model_name = 'gpt2-xl'
# great_list = ['1001_Pereira_',f'_{model_name}_lr1e-4_b8_pre10']
# # model_name = 'llama-7b'
# # great_list = ['09_30_Pereira_',f'_{model_name}_lr1e-4_b8_pre10_a0.005_l20.5']
# for user in user_list:
#     file_name = great_list[0] + user + great_list[1]
#     # os.system(f'mv ../results/{file_name}/* ../results/pre_09_30/{file_name}')
#     os.system(f'cp -r ../results/{file_name} ../final/pereira_{model_name}')

# user_list = ['P01','M04', 'M07', 'M15', 'M02']
# model_list = ['gpt2-xl','gpt2','gpt2-medium','gpt2-large','llama-7b']
# for model in model_list:
#     for user in user_list:
#         os.system(f'mv ../results/09_30_Pereira_{user}_{model}_lr1e-4_b8_pre10_a0.005_l20.5_shuffle ../results/09_30_Pereira_{user}_{model}_lr1e-4_b8_pre10_a0.005_l20.5_bl_shuffle')

import json
dirs = os.listdir('../results')
for dir in dirs:
    if 'bl' in dir:
        continue
    if os.path.exists(f'../results/{dir}/info.json'):
        log_file = json.load(open(f'../results/{dir}/info.json','r'))
        if log_file['brain_model'] == 'linear':
            print(dir)