#!/bin/sh

module purge
module load anaconda
module load compilers/cuda/11.7 cudnn/8.4.0.27_cuda11.x compilers/gcc/12.2.0 llvm/triton-llvm_17.0.0
source activate torch2.0.1_cuda117
cd ~/fmri/Brain-language-generation/end2end_generation/src
# for llama-7b
python main.py -task_name $1 -cuda 0 -load_check_point False -model_name $2 -checkpoint_path $3 -wandb none -mode evaluate -pos True -data_spliting end2end -mode end2end -use_bad_words_ids False -ncontext 10 -gcontext 10 -length_penalty 1.0 -output test.n10.g10.l1.0 -num_steps 100 
# for gpt2-xl
# python main.py -task_name $1 -cuda 0 -load_check_point False -model_name $2 -checkpoint_path $3 -wandb none -mode evaluate -pos True -data_spliting end2end -mode end2end -use_bad_words_ids False -ncontext 10 -gcontext 10 -length_penalty 0.3 -output test.n10.g10.l0.3 -num_steps 100 
# for huth
# python main.py -task_name $1 -cuda 0 -load_check_point False -model_name $2 -checkpoint_path $3 -wandb none -mode evaluate -pos True -data_spliting end2end -mode end2end -use_bad_words_ids False -ncontext 5 -gcontext 5 -output test.n5.g5 -num_steps 100 

# python main.py -task_name Huth_1 -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path Huth_1_llama-7b_ds -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode evaluate -pos True -data_spliting end2end -mode end2end

# python main.py -task_name Huth_1 -cuda 0 -load_check_point False -model_name gpt2-xl -checkpoint_path Huth_1_gpt2-xl_pre0 -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode evaluate -pos True -data_spliting end2end -mode end2end


