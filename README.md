# DUET
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/DUET/blob/main/licence)
[![arxiv badge](https://img.shields.io/badge/arxiv-2207.01328-red)](https://arxiv.org/abs/2207.01328)
 - [*DUET: Cross-modal Semantic Grounding for Contrastive Zero-shot Learning*](https://arxiv.org/abs/2207.01328)

>In this paper, we present a transformer-based end-to-end ZSL method named DUET, which integrates latent semantic knowledge from the pre-trained language models (PLMs) via a self-supervised multi-modal learning paradigm. Specifically, we **(1)** developed a cross-modal semantic grounding network to 
investigate the model's capability of disentangling semantic attributes from the images; **(2)** applied an attribute-level contrastive learning strategy to further enhance the model's discrimination on fine-grained visual characteristics against the attribute co-occurrence and imbalance; **(3)** proposed a multi-task learning policy for considering multi-model objectives.

## Model Architecture
![Model_architecture](https://github.com/zjukg/DUET/blob/main/figure/duet.png)

## Dependencies

- Python 3
- [PyTorch](http://pytorch.org/) (>= 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (**version 3.0.2**)

### Train
#### Our code will be available soon.
```shell
bash run.sh
```
### Test

```shell
bash run_test.sh
```

**Note**: 
- you can open the `.sh` file for parameter modification.

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
