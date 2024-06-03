# ActionFormer: Localizing Moments of Actions with Transformers

## Introduction
This code repo implements Actionformer, one of the first Transformer-based model for temporal action localization --- detecting the onsets and offsets of action instances and recognizing their action categories. Without bells and whistles, ActionFormer achieves 71.0% mAP at tIoU=0.5 on THUMOS14, outperforming the best prior model by 14.1 absolute percentage points and crossing the 60% mAP for the first time. Further, ActionFormer demonstrates strong results on ActivityNet 1.3 (36.56% average mAP) and the more challenging EPIC-Kitchens 100 (+13.5% average mAP over prior works). Our paper is accepted to ECCV 2022 and an arXiv version can be found at [this link](https://arxiv.org/abs/2202.07925).

In addition, ActionFormer is the backbone for many winning solutions in the Ego4D Moment Queries Challenge 2022. Our submission in particular is ranked 2nd with a record 21.76% average mAP and 42.54% Recall@1x, tIoU=0.5, nearly three times higher than the official baseline. An arXiv version of our tech report can be found at [this link](https://arxiv.org/abs/2211.09074). We invite our audience to try out the code.

# Requirements

- Linux
- Python 3.5+
- PyTorch 1.11
- TensorBoard
- CUDA 11.0+
- GCC 4.9+
- 1.11 <= Numpy <= 1.23
- PyYaml
- Pandas
- h5py
- joblib

# Compilation

Part of NMS is implemented in C++. The code can be compiled by

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

The code should be recompiled every time you update PyTorch.

# Data Prepration
Put the annotation files (`proficiency_demonstration_train.json`, `proficiency_demonstration_val.json`, 
`proficiency_demonstration_test.json`, `takes.json`) in one directory and run

```
python -m generate_thumos_export path_to_folder arxivb 
```



