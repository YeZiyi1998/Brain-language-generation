#!/bin/sh

module purge
module load anaconda
module load compilers/cuda/11.7 cudnn/8.4.0.27_cuda11.x compilers/gcc/12.2.0 llvm/triton-llvm_17.0.0
source activate torch2.0.1_cuda117
cd ~/fmri/Brain-language-generation/word_rate/src
python main.py -task_name Huth_1 -checkpoint_path Huth_1 -wandb none -mode all
python main.py -task_name Huth_2 -checkpoint_path Huth_2 -wandb none -mode all
python main.py -task_name Huth_3 -checkpoint_path Huth_3 -wandb none -mode all
# python main.py -task_name Huth_4 -checkpoint_path Huth_4 -wandb none -mode all
# python main.py -task_name Huth_5 -checkpoint_path Huth_5 -wandb none -mode all
# python main.py -task_name Huth_6 -checkpoint_path Huth_6 -wandb none -mode all
# python main.py -task_name Huth_7 -checkpoint_path Huth_7 -wandb none -mode all
# python main.py -task_name Huth_7 -checkpoint_path Huth_8 -wandb none -mode all
