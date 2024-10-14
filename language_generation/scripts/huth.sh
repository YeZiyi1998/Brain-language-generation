#!/bin/sh

module purge
module load anaconda
module load compilers/cuda/11.7 cudnn/8.4.0.27_cuda11.x compilers/gcc/12.2.0 llvm/triton-llvm_17.0.0
source activate torch2.0.1_cuda117
cd ~/fmri/Brain-language-generation/language_generation/src
python main.py -task_name $1 -cuda 0 -load_check_point False -model_name $2 -checkpoint_path $3 -batch_size 8 -lr 1e-4 -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode all -pos True -early_stop 5 -mode train 
python main.py -task_name $1 -cuda 0 -load_check_point False -model_name $2 -checkpoint_path $3 -batch_size 8 -lr 1e-4 -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode all -pos True -mode evaluate 
python main.py -task_name $1 -cuda 0 -load_check_point False -model_name $2 -checkpoint_path $3 -batch_size 8 -lr 1e-4 -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode all -pos True -input_method permutated -mode evaluate -output test_permutated
