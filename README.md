## Molecular Contrastive Learning of Representations via Graph Neural Networks ##

#### Nature Machine Intelligence [[Paper]](https://www.nature.com/articles/s42256-022-00447-x) [[arXiv]](https://arxiv.org/abs/2102.10056/) [[PDF]](https://www.nature.com/articles/s42256-022-00447-x.pdf) </br>
[Yuyang Wang](https://yuyangw.github.io/), [Jianren Wang](https://www.jianrenw.com/), [Zhonglin Cao](https://www.linkedin.com/in/zhonglincao/?trk=public_profile_browsemap), [Amir Barati Farimani](https://www.meche.engineering.cmu.edu/directory/bios/barati-farimani-amir.html) </br>
Carnegie Mellon University </br>

<img src="figs/pipeline.gif" width="450">

This is the official implementation of <strong><em>MolCLR</em></strong>: ["Molecular Contrastive Learning of Representations via Graph Neural Networks"](https://www.nature.com/articles/s42256-022-00447-x). In this work, we introduce a contrastive learning framework for molecular representation learning on large unlabelled dataset (~10M unique molecules). <strong><em>MolCLR</em></strong> pre-training greatly boosts the performance of GNN models on various downstream molecular property prediction benchmarks. 
If you find our work useful in your research, please cite:

```
@article{wang2022molclr,
  title={Molecular contrastive learning of representations via graph neural networks},
  author={Wang, Yuyang and Wang, Jianren and Cao, Zhonglin and Barati Farimani, Amir},
  journal={Nature Machine Intelligence},
  pages={1--9},
  year={2022},
  publisher={Nature Publishing Group}
}
```


## Getting Started

### Installation

Set up conda environment and clone the github repo

```
# create a new environment
$ conda create --name molclr python=3.7
$ conda activate molclr

# install requirements
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
$ pip install PyYAML
$ conda install -c conda-forge rdkit tensorboard
$ conda install -c conda-forge nvidia-apex # optional

# clone the source code of MolCLR
$ git clone https://github.com/yuyangw/MolCLR.git
$ cd MolCLR
```

### Dataset

You can download the pre-training data and benchmarks used in the paper [here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view?usp=sharing) and extract the zip file under `./data` folder. The data for pre-training can be found in `pubchem-10m-clean.txt`. All the databases for fine-tuning are saved in the folder under the benchmark name. You can also find the benchmarks from [MoleculeNet](https://moleculenet.org/).

### Pre-training

To train the MolCLR, where the configurations and detailed explaination for each variable can be found in `config.yaml`
```
$ python molclr.py
```

To monitor the training via tensorboard, run `tensorboard --logdir ckpt/{PATH}` and click the URL http://127.0.0.1:6006/.

### Fine-tuning 

To fine-tune the MolCLR pre-trained model on downstream molecular benchmarks, where the configurations and detailed explaination for each variable can be found in `config_finetune.yaml`
```
$ python finetune.py
```

### Pre-trained models

We also provide pre-trained GCN and GIN models, which can be found in `ckpt/pretrained_gin` and `ckpt/pretrained_gcn` respectively. 

## Acknowledgement

- PyTorch implementation of SimCLR: [https://github.com/sthalles/SimCLR](https://github.com/sthalles/SimCLR)
- Strategies for Pre-training Graph Neural Networks: [https://github.com/snap-stanford/pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns)