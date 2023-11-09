# DUET
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/DUET/blob/main/licence)
[![arxiv badge](https://img.shields.io/badge/arxiv-2207.01328-red)](https://arxiv.org/abs/2207.01328)
[![AAAI](https://img.shields.io/badge/AAAI-2023-%23f1592a?labelColor=%23003973&color=%23be1c1a)](https://aaai.org/Conferences/AAAI-23/)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
 - [*DUET: Cross-modal Semantic Grounding for Contrastive Zero-shot Learning*](https://arxiv.org/abs/2207.01328)

>In this paper, we present a transformer-based end-to-end ZSL method named DUET, which integrates latent semantic knowledge from the pre-trained language models (PLMs) via a self-supervised multi-modal learning paradigm. Specifically, we **(1)** developed a cross-modal semantic grounding network to 
investigate the model's capability of disentangling semantic attributes from the images; **(2)** applied an attribute-level contrastive learning strategy to further enhance the model's discrimination on fine-grained visual characteristics against the attribute co-occurrence and imbalance; **(3)** proposed a multi-task learning policy for considering multi-model objectives.

- Due to the **```page and format restrictions```** set by AAAI publications, we have omitted some details and appendix content. For the complete version of the paper, including the **```selection of prompts```** and **```experiment details```**, please refer to our [arXiv version](https://arxiv.org/abs/2207.01328).

## 🤖 Model Architecture
![Model_architecture](https://github.com/zjukg/DUET/blob/main/figure/duet.png)

## 📚 Dataset Download
- The cache data for **`(CUB, AWA, SUN)`** are available [`here`](https://pan.baidu.com/s/13oyLDNm6uoYpVgcMitrY-A) (`Baidu cloud`, **`19.89G`**, Code: `s07d`).

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

## 🔬 Dependencies

- ```Python 3```
- ```PyTorch >= 1.8.0```
- ```Transformers>= 4.11.3```
- ```NumPy```
- All experiments are performed with one RTX 3090Ti GPU.

## 🎯  Prerequisites
- **Dataset**: please download the dataset, i.e., [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [AWA2](https://cvml.ist.ac.at/AwA2/), [SUN](https://cs.brown.edu/~gmpatter/sunattributes.html), and change the ```opt.image_root``` to the dataset root path on your machine
  - ❗NOTE: For other required feature files like `APN.mat` and `id2imagepixel.pkl`, **please refer to [here](https://github.com/zjukg/DUET/issues/2)**.
- **Data split**: please download the data folder and place it in ```./data/```.
- ```Attributeindex2prompt.json``` should generate and place it in ```./cache/dataset/```.
- Download pretrained vision Transformer as the vision encoder: 
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

📌 **Note**: 
- you can open the `.sh` file for <a href="#Parameter">parameter</a> modification.
- Don't worry if you have any question. Just feel free to let we know via **`Adding Issues`**.

## 🤝 Cite:
Please consider citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```bigquery
@inproceedings{DBLP:conf/aaai/ChenHCGZFPC23,
  author       = {Zhuo Chen and
                  Yufeng Huang and
                  Jiaoyan Chen and
                  Yuxia Geng and
                  Wen Zhang and
                  Yin Fang and
                  Jeff Z. Pan and
                  Huajun Chen},
  title        = {{DUET:} Cross-Modal Semantic Grounding for Contrastive Zero-Shot Learning},
  booktitle    = {{AAAI}},
  pages        = {405--413},
  publisher    = {{AAAI} Press},
  year         = {2023}
}
```
<a href="https://info.flagcounter.com/VOlE"><img src="https://s11.flagcounter.com/count2/VOlE/bg_FFFFFF/txt_000000/border_F7F7F7/columns_6/maxflags_12/viewers_3/labels_0/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
