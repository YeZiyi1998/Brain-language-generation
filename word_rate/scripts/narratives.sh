#!/bin/sh

module purge
module load anaconda
module load compilers/cuda/11.7 cudnn/8.4.0.27_cuda11.x compilers/gcc/12.2.0 llvm/triton-llvm_17.0.0
source activate torch2.0.1_cuda117
cd ~/fmri/Brain-language-generation/word_rate/src
python main.py -task_name Narratives_016 -checkpoint_path Narratives_016 -wandb none -mode all
python main.py -task_name Narratives_052 -checkpoint_path Narratives_052 -wandb none -mode all
python main.py -task_name Narratives_065 -checkpoint_path Narratives_065 -wandb none -mode all
