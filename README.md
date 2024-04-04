



# Unifying Correspondence, Pose and NeRF for Generalized Pose-Free Novel View Synthesis from Stereo Pairs [CVPR 2024]
This is our official implementation of CoPoNeRF! 

[[arXiv](https://arxiv.org/pdf/2312.07246)] [[Project](https://ku-cvlab.github.io/CoPoNeRF/)] <br>

by [Sunghwan Hong](https://sunghwanhong.github.io/), [Jaewoo Jung](https://crepejung00.github.io/), [Heeseong Shin](https://github.com/hsshin98), [Jiaolong Yang](https://jlyang.org/), [Seungryong Kim](https://cvlab.korea.ac.kr), [Chong Luo](https://www.microsoft.com/en-us/research/people/cluo/?from=https://research.microsoft.com/en-us/people/cluo/&type=exacthttps://www.microsoft.com/en-us/research/people/cluo/?from=https://research.microsoft.com/en-us/people/cluo/&type=exact), 

## Introduction
![](assets/main_architecture.png)
We delve into the task of generalized pose-free novel view synthesis from stereo pairs, a challenging and pioneering task in 3D vision.

For further details and visualization results, please check out our [paper](https://arxiv.org/pdf/2312.07246) and our [project page](https://ku-cvlab.github.io/CoPoNeRF/).

**❗️Update:** This repository includes refactored codes:
- We retrained the network using this code base. 
- If you want the codes and weights for the CVPR version, please email Sunghwan!


## Installation

### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.13 is recommended and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. 

An example of installation is shown below:

```
git clone https://github.com/KU-CVLAB/CoPoNeRF.git
cd CoPoNeRF
conda create -n CoPoNeRF python=3.8
conda activate CoPoNeRF
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r assets/requirements.txt
```

## Data Preparation
Please follow [dataset preperation](data_download/README.md).


## Training
``` 
python train.py --experiment_name [EXPERIMENT_NAME] 
```

If you want to specify batch size or number of gpus, include them as arguments. Also, you can freely add the losses, e.g., cycle loss. To train on ACID, you can do it by simply changing the directories and the dataset in train.py. 

## Evaluation

```
python test.py --checkpoint_path [CHECKPOINT_PATH]
```

## Pretrained Models
We provide pretrained weights [here](https://drive.google.com/file/d/1z97TGEIIGeZtqt_a2smuWWmKQjRmnUJQ/view?usp=drive_link). These models were trained with 4 A6000. 


## Acknowledgement
We would like to acknowledge the contributions of public projects, such as [Du et al.](https://github.com/yilundu/cross_attention_renderer#get-started) and [UFC](https://github.com/KU-CVLAB/UFC), whose code has been utilized in this repository.
## Citing CoPoNeRF:

```BibTeX
@article{hong2023unifying,
  title={Unifying Correspondence, Pose and NeRF for Pose-Free Novel View Synthesis from Stereo Pairs},
  author={Hong, Sunghwan and Jung, Jaewoo and Shin, Heeseong and Yang, Jiaolong and Kim, Seungryong and Luo, Chong},
  journal={arXiv preprint arXiv:2312.07246},
  year={2023}
}

