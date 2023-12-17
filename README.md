# Language Generation from Brain Recordings

<img src="./figures/model_structure2.jpg" alt="Logo" height="80%">

<p align="left">
    <a href="https://github.com/YeZiyi1998/Brain-language-generation">
    <img alt="BCI" src="https://img.shields.io/badge/BCI-Language%20Generation-blueviolet">
    </a>
    <a href="https://github.com/YeZiyi1998/Brain-language-generation/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
    <a>
    <a href="https://pytorch.org">
    <img alt="made-with-pytorch" src="https://img.shields.io/badge/Made%20with-Pytorch-red.svg">
    </a>
    <a>
    <a href="https://github.com/YeZiyi1998/Brain-language-generation">
    <img alt="code-size" src="https://img.shields.io/github/languages/code-size/YeZiyi1998/Brain-language-generation?color=green">
    </a>
    <a href="https://github.com/YeZiyi1998/Brain-language-generation">
    </a>
</p>

This is the official repo for our paper [Language Generation from Brain Recordings](https://arxiv.org/abs/2311.09889). Language generation from brain recordings is a novel approach that supports direct language generation with BCIs (brain-computer interfaces) without pre-defineng or pre-generating language candidates to select from.


## Quick Start
We have provided a example dataset to facilitate the replication of experiments. To run the example dataset, you can go into the sub-directory language_generation/src and use the following command:

```bash
# model training and evaluation (runing BrainLLM)
python main.py -task_name Pereira_example -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path example -batch_size 8 -lr 1e-4 -pos False -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode all
# control evaluation (runing PerBrainLLM)
python main.py -task_name Pereira_example -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path example -batch_size 8 -lr 1e-4 -pos False -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -input_method permutated -mode evaluate -output test_permutated
# control evaluation (runing LLM)
python main.py -task_name Pereira_example -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path example -batch_size 8 -lr 1e-4 -pos False -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -input_method mask_input -mode evaluate -output test_nobrain
```

To run with slurms, you can also use the 

### Installation

This repo is developed with [PyTorch]((https://pytorch.org/get-started/locally/)). It can be installed manually according to the requirement of platform-specific custom configuration. The recommended commands for installation are:
```bash
# XX.X is a placeholder for cudatoolkit version. It should be specified according to your environment
conda install pytorch torchvision torchaudio cudatoolkit=XX.X -c pytorch 
```
In our experiment, we use torch verison 2.0.1 and cuda verison 11.7.
In addition to PyTorch, we adopt several publicly available packages, which can be installed by
```bash
pip install -r requirements.txt
```

### Model Training


### Model Evaluation


### Dataset



## Citation
If you find our work helpful, please consider citing us:
```
@article{ye2023language,
  title={Language Generation from Human Brain Activities},
  author={Ye, Ziyi and Ai, Qingyao and Liu, Yiqun and Zhang, Min and Lioma, Christina and Ruotsalo, Tuukka},
  journal={arXiv preprint arXiv:2311.09889},
  year={2023}
}
```