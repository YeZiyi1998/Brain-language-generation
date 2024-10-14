#!/bin/sh

module purge
module load anaconda
module load compilers/cuda/11.7 cudnn/8.4.0.27_cuda11.x compilers/gcc/12.2.0 llvm/triton-llvm_17.0.0
source activate torch2.0.1_cuda117
cd ~/fmri/Brain-language-generation/end2end_generation/src

python main.py -task_name $1 -cuda 0 -load_check_point False -model_name $2 -checkpoint_path $3 -wandb none -mode evaluate -pos True -data_spliting end2end -mode end2end -use_bad_words_ids False -ncontext 10 -gcontext 10 -length_penalty 0.3 -beam_width 3 -extensions 3
