#!/bin/sh

module purge
module load anaconda
module load compilers/cuda/11.7 cudnn/8.4.0.27_cuda11.x compilers/gcc/12.2.0 llvm/triton-llvm_17.0.0
source activate torch2.0.1_cuda117
cd ~/fmri/Brain-language-generation/word_rate/src
python main.py -task_name Pereira_P01 -checkpoint_path Pereira_P01 -wandb none -mode all
python main.py -task_name Pereira_M02 -checkpoint_path Pereira_M02 -wandb none -mode all
python main.py -task_name Pereira_M04 -checkpoint_path Pereira_M04 -wandb none -mode all
python main.py -task_name Pereira_M07 -checkpoint_path Pereira_M07 -wandb none -mode all
python main.py -task_name Pereira_M15 -checkpoint_path Pereira_M15 -wandb none -mode all
