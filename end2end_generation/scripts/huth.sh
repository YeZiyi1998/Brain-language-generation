#!/bin/sh

module purge
module load anaconda
module load compilers/cuda/11.7 cudnn/8.4.0.27_cuda11.x compilers/gcc/12.2.0 llvm/triton-llvm_17.0.0
source activate torch2.0.1_cuda117
cd ~/fmri/Brain-language-generation/end2end_generation/src
# python main.py -task_name Huth_1 -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path Huth_1 -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode evaluate -pos True -data_spliting end2end -mode end2end
# python main.py -task_name Huth_1 -cuda 0 -load_check_point False -model_name gpt2-xl -checkpoint_path Huth_1_gpt2-xl -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode evaluate -pos True -data_spliting end2end -mode end2end
python main.py -task_name Huth_1 -cuda 0 -load_check_point False -model_name gpt2-xl -checkpoint_path Huth_1_gpt2-xl_grad -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode evaluate -pos True -data_spliting end2end -mode end2end -use_bad_words_ids True 

