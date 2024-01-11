#!/bin/sh

module purge
module load anaconda
module load compilers/cuda/11.7 cudnn/8.4.0.27_cuda11.x compilers/gcc/12.2.0 llvm/triton-llvm_17.0.0
source activate torch2.0.1_cuda117
cd ~/fmri/Brain-language-generation/language_generation/src
python main.py -task_name Narratives_016 -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path Narratives_016 -batch_size 8 -lr 1e-4 -pos False -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode all -dataset_path ../../../dataset/preprocessed/Narratives/ -pos True
python main.py -task_name Narratives_016 -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path Narratives_016 -batch_size 8 -lr 1e-4 -pos False -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -input_method permutated -mode evaluate -output test_permutated -dataset_path ../../../dataset/preprocessed/Narratives/ -pos True
python main.py -task_name Narratives_016 -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path Narratives_016 -batch_size 8 -lr 1e-4 -pos False -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -input_method without_brain -mode evaluate -output test_nobrain -dataset_path ../../../dataset/preprocessed/Narratives/ -pos True