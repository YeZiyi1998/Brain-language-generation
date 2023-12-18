# Language Generation from Brain Recordings

<img src="./figures/model_structure2.jpg" alt="Logo" height="80%">

<p align="left">
    <a href="https://github.com/YeZiyi1998/Brain-language-generation">
    <img alt="BCI" src="https://img.shields.io/badge/BCI-Language%20Generation-blueviolet">
    </a>
    <a href="https://github.com/YeZiyi1998/Brain-language-generation/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-'cc by-nc 4.0'-blue.svg">
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
We have provided an example dataset to facilitate the replication of experiments. To run the example dataset, you can go into the sub-directory *language_generation/src* and use the following command:

```bash
# model training and evaluation (runing BrainLLM)
python main.py -task_name Pereira_example -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path example -batch_size 8 -lr 1e-4 -pos False -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode all
# control evaluation (runing PerBrainLLM)
python main.py -task_name Pereira_example -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path example -batch_size 8 -lr 1e-4 -pos False -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -input_method permutated -mode evaluate -output test_permutated
# control evaluation (runing LLM)
python main.py -task_name Pereira_example -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path example -batch_size 8 -lr 1e-4 -pos False -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -input_method mask_input -mode evaluate -output test_nobrain
```

To run with [slurm](https://slurm.schedmd.com/documentation.html), you can also use the provided scripts in the sub-directory *language_generation/scripts* (remember to replace the name of conda environment and the path of the sub-directory *language_generation/scripts* according to your settings).

```bash
sh example.sh
```

To run with the datasets utilized in our paper, please download the dataset from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/04e8cfe6c9c743c69f08/) and unzip it. Use the parameter *-dataset_path* to specify the path of your unzip dataset.
For example, if you unzip the dataset into your home directory as *~/released/*, then you can run the training and evaluation of BrainLLM and the participant 1 in Huth dataset using the following command:
```bash
python main.py -task_name Huth_1 -cuda 0 -load_check_point False -model_name llama-7b -checkpoint_path Huth_1 -batch_size 8 -lr 1e-4 -pos False -pretrain_lr 1e-3 -pretrain_epochs 10 -wandb none -mode all -dataset_path ~/released/Huth/ -pos True
``` 

To evaluate the model performance, you can refer to the code in *language_generation/src/post_hoc_evaluate.py*

### Installation

This repo is developed with [PyTorch](https://pytorch.org/get-started/locally/). It can be installed manually according to the requirement of platform-specific custom configuration. The recommended commands for installation are:
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
To train the model, you need to special the parameter *-mode* as *training* (only training) or *all* (training and evaluation).
You can specify several hyper parameters according to your requirement, the default parameters for Pereira's dataset, Huth's dataset, and Narratives dataset are provided in *language_generation/scripts/example.sh*, *language_generation/scripts/huth.sh*, and *language_generation/scripts/narratives.sh*, respectively.
The meaning of hyper parameters are listed below:

|  **Parameter**  | **Meaning**  |
|   :----   |   :----   |
| model_name | the selected LLM, choose from {gpt2,gpt2-medium,gpt2-large,gpt2-xl,llama-2} |
| method | only supported *decoding* in the released verison |
| task_name | *{dataset_name}_{participant_name}*, dataset_name selected from *{Pereira,Huth,Narratives}* |
| test_trail_ids | specify the range of test dataset, view the dict *dataset2agrs* in *language_generation/src/config.py* for default setting |
| valid_trail_ids | specify the range of validation dataset, view the dict *dataset2agrs* in *language_generation/src/config.py* for default setting |
| random_number | for cross-validation evaluation, cooperate with parameter *test_trail_ids* and *valid_trail_ids*|
| batch_size | set as 8 in our experiment |
| fmri_pca | how to do data dimensionality reduction, default is *True* |
| cuda | specify the device number |
| layer | not used in the released verison |
| num_epochs | specify the maximum number of training epochs |
| lr | learning rate, set as 1e-4 in our experiment |
| dropout | dropout rate for brain decoder |
| checkpoint_path | path of training checkpoint for saving and downloading |
| load_check_point | whether to load existing checkpoint |
| enable_grad | whether to allow the parameter in LLM updated or not |
| mode | *train*: only training and evaluate in the validation set; *evaluate*: evaluate in the test set; *all*: train and evaluate|
| additional_loss | training with additional loss, not used in the released verison |
| fake_input | training with fake input, not used in the released verison |
| add_end | not used in the released verison |
| context | whether to discard data sample without any text prompt or not |
| roi_selected | roi-based experiment, not used in the released verison |
| project_name | specify the project name for [wandb](https://wandb.ai/site) |
| noise_ratio | not used in the released verison |
| wandb | specify how to sync the experimental in [wandb](https://wandb.ai/site), selected from *{online, offline, none}* |
| generation_method | generation method for the LLM, selected from *{greeddy, beam}* |
| pos | specify whether to use position embedding in the brain decoder |
| output | specify whether to use position embedding in the brain decoder |
| data_spliting | specify how to split the dataset, selected from *{random, cross_story}*, default is *random* |
| brain_model | the based model for the brain decoder, selected from *{mlp,rnn,linear,big_mlp,multi_mlp}* |
| weight_decay | weight decay |
| l2 | weight for l2 regularized loss |
| num_layers | number of layers in the brain decoder |
| evaluate_log | whether to evaluate in the test set for model in each training epoch |
| normalized | whether to normalize the input |
| activation | activation function, selected from *{relu,sigmoid,tanh,relu6}* |
| pretrain_epochs | number of epochs in warm up step |
| pretrain_lr | learning rate in warm up step|
| data_size | maximum training data samples |
| results_path | path to save model results |
| dataset_path | path to the downloaded dataset |
| shuffle_times | permutation times for PerBrainLLM |

### Model Evaluation
To evaluate the model with different prompt input, i.e., BrainLLM, PerBrainLLM, and LLM, you can specify the parameter *-input_method* as *normal*, *permutated*, *without_brain*, respectively. To test the model performance without any text prompt, you should train and evaluate the model while setting *-input_method* as *without_text*.

After that, you can get output files for different prompt inputs. Then, you can evaluate their performance by runing the python script *language_generation/src/post_hoc_evaluatoion.py* with the path of output files specified.
Refer to *language_generation/src/post_hoc_evaluatoion.py* for example usage:
```bash
python language_generation/src/post_hoc_evaluatoion.py
```

### Dataset
We test our approach on three public fMRI datasets: [Pereira's dataset](https://www.nature.com/articles/s41467-018-03068-4), [Huth's dataset](https://www.nature.com/articles/s41597-023-02437-z), and [Narratives dataset](https://www.nature.com/articles/s41597-021-01033-3). The brief introduction, ethical information, statistics, and useage details of these datasets are provied in our paper.
A preprocessed verison dataset is released in [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/04e8cfe6c9c743c69f08/), where the sub-directory of *Pereira*, *Huth*, and *Narratives* contain the preprocessed data for each participant and story in Pereira's dataset, Huth's dataset, and Narratives dataset, respectively. 

### Experimental results
This is the overall experimental results in terms of language similarity metrics. Refer to our paper for the explaination of metrics and more analyses.
| Dataset    | Model        | Bleu-1(↑) | ROUGE-1(↑) | ROUGE-L(↑) | WER(↓) |
|------------|--------------|-----------|------------|------------|--------|
| Pereira’s  | BrainLLM     | 0.3333    | 0.2987     | 0.2877     | 0.7681 |
|            | PerBrainLLM  | 0.3249    | 0.2875     | 0.2771     | 0.7781 |
|            | LLM          | 0.2415    | 0.2133     | 0.2096     | 0.8349 |
| Huth’s     | BrainLLM     | 0.1899    | 0.1780     | 0.1709     | 0.8946 |
|            | PerBrainLLM  | 0.1668    | 0.1536     | 0.1474     | 0.9109 |
|            | LLM          | 0.1500    | 0.1360     | 0.1310     | 0.9200 |
| Narratives | BrainLLM     | 0.1375    | 0.1249     | 0.1209     | 0.9239 |
|            | PerBrainLLM  | 0.1269    | 0.1144     | 0.1105     | 0.9311 |
|            | LLM          | 0.0953    | 0.0858     | 0.0829     | 0.9485 |


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