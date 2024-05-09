#!/bin/sh

module purge
module load anaconda
module load compilers/cuda/11.7 cudnn/8.4.0.27_cuda11.x compilers/gcc/12.2.0 llvm/triton-llvm_17.0.0
source activate torch2.0.1_cuda117
cd ~/fmri/Brain-language-generation/word_rate/src
# python main.py -task_name Huth_1 -checkpoint_path Huth_1 -wandb none -mode all -data_spliting end2end -model huth
python main.py -task_name Huth_2 -checkpoint_path Huth_2_huth -wandb none -mode all -data_spliting end2end -model huth
python main.py -task_name Huth_3 -checkpoint_path Huth_3_huth -wandb none -mode all -data_spliting end2end -model huth
python main.py -task_name Huth_1 -checkpoint_path Huth_1_gpt2-xl -wandb none -mode all -data_spliting end2end -model gpt2-xl
python main.py -task_name Huth_2 -checkpoint_path Huth_2_gpt2-xl -wandb none -mode all -data_spliting end2end -model gpt2-xl
python main.py -task_name Huth_3 -checkpoint_path Huth_3_gpt2-xl -wandb none -mode all -data_spliting end2end -model gpt2-xl
python main.py -task_name Huth_1 -checkpoint_path Huth_1_llama-7b -wandb none -mode all -data_spliting end2end -model llama-7b
python main.py -task_name Huth_2 -checkpoint_path Huth_2_llama-7b -wandb none -mode all -data_spliting end2end -model llama-7b
python main.py -task_name Huth_3 -checkpoint_path Huth_3_llama-7b -wandb none -mode all -data_spliting end2end -model llama-7b