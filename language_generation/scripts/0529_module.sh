#!/bin/sh

module purge
module load anaconda
module load compilers/cuda/11.7 cudnn/8.4.0.27_cuda11.x compilers/gcc/12.2.0 llvm/triton-llvm_17.0.0
source activate torch2.0.1_cuda117
cd ~/fmri/Brain-language-generation/language_generation/src
python main.py -task_name Huth_1 -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path $1 -batch_size 8 -lr 1e-4 -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode all -pos True -early_stop 5 -mode train -brain_model $2
python main.py -task_name Huth_1 -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path $1 -batch_size 8 -lr 1e-4 -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode all -pos True -mode evaluate -brain_model $2
python main.py -task_name Huth_1 -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path $1 -batch_size 8 -lr 1e-4 -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode all -pos True -input_method permutated -mode evaluate -output test_permutated -brain_model $2


# multi_mlp, big_mlp, mlp, linear, rnn