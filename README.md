# DUET
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/DUET/blob/main/licence)
[![arxiv badge](https://img.shields.io/badge/arxiv-2207.01328-red)](https://arxiv.org/abs/2207.01328)
[![AAAI](https://img.shields.io/badge/AAAI'23-brightgreen)](https://aaai.org/Conferences/AAAI-23/)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
 - [*DUET: Cross-modal Semantic Grounding for Contrastive Zero-shot Learning*](https://arxiv.org/abs/2207.01328)

>In this paper, we present a transformer-based end-to-end ZSL method named DUET, which integrates latent semantic knowledge from the pre-trained language models (PLMs) via a self-supervised multi-modal learning paradigm. Specifically, we **(1)** developed a cross-modal semantic grounding network to 
investigate the model's capability of disentangling semantic attributes from the images; **(2)** applied an attribute-level contrastive learning strategy to further enhance the model's discrimination on fine-grained visual characteristics against the attribute co-occurrence and imbalance; **(3)** proposed a multi-task learning policy for considering multi-model objectives.

## 🌈 Model Architecture
![Model_architecture](https://github.com/zjukg/DUET/blob/main/figure/duet.png)

## 📕 Code Path

#### Code Structures
There are four parts in the code.
- **model**: It contains the main files for DUET network.
- **data**: It contains the data splits for different datasets.
- **cache**: It contains some cache files.
- **script**: The training scripts for DUET.

```shell
DUET
├── cache
│   ├── AWA2
│   │   ├── attributeindex2prompt.json
│   │   └── id2imagepixel.pkl
│   ├── CUB
│   │   ├── attributeindex2prompt.json
│   │   ├── id2imagepixel.pkl
│   │   └── mapping.json
│   └── SUN
│   │   ├── attributeindex2prompt.json
│   │   ├── id2imagepixel.pkl
│   │   └── mapping.json
├── data
│   ├── AWA2
│   │   ├── APN.mat
│   │   ├── TransE_65000.mat
│   │   ├── att_splits.mat
│   │   ├── attri_groups_9.json
│   │   ├── kge_CH_AH_CA_60000.mat
│   │   └── res101.mat
│   ├── CUB
│   │   ├── APN.mat
│   │   ├── att_splits.mat
│   │   ├── attri_groups_8.json
│   │   ├── attri_groups_8_layer.json
│   │   └── res101.mat
│   └── SUN
│       ├── APN.mat
│       ├── att_splits.mat
│       ├── attri_groups_4.json
│       └── res101.mat
├── log
│   ├── AWA2
│   ├── CUB
│   └── SUN
├── model
│   ├── log.py
│   ├── main.py
│   ├── main_utils.py
│   ├── model_proto.py
│   ├── modeling_lxmert.py
│   ├── opt.py
│   ├── swin_modeling_bert.py
│   ├── util.py
│   └── visual_utils.py
├── out
│   ├── AWA2
│   ├── CUB
│   └── SUN
└── script
    ├── AWA2
    │   └── AWA2_GZSL.sh
    ├── CUB
    │   └── CUB_GZSL.sh
    └── SUN
        └── SUN_GZSL.sh
```

## 📚 Dependencies

- ```Python 3```
- ```PyTorch >= 1.8.0```
- ```Transformers>= 4.11.3```
- ```NumPy```
- All experiments are performed with one RTX 3090Ti GPU.

## ⚡ Prerequisites
- **Dataset**: please download the dataset, i.e., [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [AWA2](https://cvml.ist.ac.at/AwA2/), [SUN](https://groups.csail.mit.edu/vision/SUN/hierarchy.html), and change the ```opt.image_root``` to the dataset root path on your machine
- **Data split**: please download the data folder and place it in ```./data/```.
- ```Attributeindex2prompt.json``` should generate and place it in ```./cache/dataset/```.
- Download pretranined vision Transformer as the vision encoder: 
  - [deit-base-distilled-patch16-224](https://huggingface.co/facebook/deit-base-distilled-patch16-224)
  - [swin_base_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth) 



## 🚀 Train & Eval

The training script for **AWA2_GZSL**:
```shell
bash script/AWA2/AWA2_GZSL.sh
```

### [Parameter](#content)
```
[--dataset {AWA2, SUN, CUB}] [--calibrated_stacking CALIBRATED_STACKING] [--nepoch NEPOCH] [--batch_size BATCH_SIZE] [--manualSeed MANUAL_SEED]
[--classifier_lr LEARNING-RATE] [--xe XE] [--attri ATTRI] [--gzsl] [--patient PATIENT] [--model_name MODEL_NAME] [--mask_pro MASK-PRO] 
[--mask_loss_xishu MASK_LOSS_XISHU] [--xlayer_num XLAYER_NUM] [--construct_loss_weight CONSTRUCT_LOSS_WEIGHT] [--sc_loss SC_LOSS] [--mask_way MASK_WAY]
[--attribute_miss ATTRIBUTE_MISS]
```

**Note**: 
- you can open the `.sh` file for <a href="#Parameter">parameter</a> modification.

## Cite:
Please condiser citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```bigquery
@InProceedings{Chen2023DUET,
    author    = {Chen, Zhuo and Huang, Yufeng and Chen, Jiaoyan and Geng, Yuxia and Zhang, Wen and Fang, Yin and Pan, Jeff Z and Song, Wenting and Chen, Huajun},
    title     = {DUET: Cross-modal Semantic Grounding for Contrastive Zero-shot Learning},
    booktitle = {Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI)},
    year      = {2023}
}
```
