Hyperbolic Conversation Network (HCN)
==================================================

## 1. Overview

This repository contains the codebase for HCN, the model introduced in the paper "Towards Suicide Ideation Detection Through Online Conversational Context" --- The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval.

## 2. Setup
### 2.1 Dependencies

```virtualenv -p [PATH to python3.8 binary] hcn```

```source hcn/bin/activate```

```pip install -r requirements.txt```

## 3. Usage

### 3.1  ```main.py```

To run the training script:

```python main.py```

This script trains HCN for classification tasks. 

```
 Arguments:
  --h-size DIM          Hidden embedding dimension
  --c C                 Curvature
  --x-size DIM          Input edimension
  --batch-size  BS      Batch size
  --data-dir DIR        Directory for data
  --device DEVICE       Device
  --lr LR               Learning rate
  --dropout DROPOUT     Dropout probability
  --epochs EPOCHS       Maximum number of epochs to train for
  --weight-decay WEIGHT_DECAY
                        L2 regularization strength
  --optimizer OPTIMIZER
                        Which optimizer to use
  --patience PATIENCE   Patience for early stopping
  --beta BETA           CB loss hyperparameter
  --gamma GAMMA         CB loss hyperparameter
  --save                Save computed results
  --save-dir SAVE_DIR   Path to save results
  --min-epochs MIN_EPOCHS
                        Do not early stop before min-epochs
```

### 3.2 Dataset Format
We describe the dataset format below. The comments for each root tweet were mined using the [Twint](https://github.com/twintproject/twint) tool. The processed dataset format should be a .pkl file having a list of trees in the form of **directed** [DGL graphs](https://docs.dgl.ai/api/python/dgl.DGLGraph.html#). For simplicity of implementation, edges must already be reversed i.e the edges must point from the child node to the parent node. 

Each node (i.e tweet) in each graph must be associated with the following attributes:

1. 'x': The 1-D embedding vector of the tweet text.
2. 'y': The associated label of the node.
3. 'del_t': The difference in timestamps between the node and its parent.
4. 'train_mask': 1 if the node (and hence, the tree it is the root of) is for training, else 0
5. 'val_mask': 1 if the node (and hence, the tree it is the root of) is for validation, else 0
6. 'test_mask': 1 if the node (and hence, the tree it is the root of) is a testing, else 0

In this work we utilize data from prior work [2]. In compliance with Twitter's privacy guidelines, and the ethical considerations discussed in prior work [2] on suicide ideation detection on social media data, we redirect researchers to the prior work that introduced the suicide ideation Twitter dataset [2] to request access to the data. We provide a sample of the processed dataset, however it is a small sample of the original dataset and hence the results obtained on this sample are not fully representative of the results that are obtained on the full dataset.

## Some of the code was forked from the following repositories
 * [geoopt](https://github.com/geoopt/geoopt)
 * [hgcn](https://github.com/HazyResearch/hgcn)

## Cite

If our work was helpful in your research, kindly consider citing us:
```
@inproceedings{10.1145/3477495.3532068,
author = {Sawhney, Ramit and Agarwal, Shivam and Neerkaje, Atula Tejaswi and Aletras, Nikolaos and Nakov, Preslav and Flek, Lucie},
title = {Towards Suicide Ideation Detection Through Online Conversational Context},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3532068},
doi = {10.1145/3477495.3532068},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1716–1727},
numpages = {12},
keywords = {social media, suicide ideation, conversation trees},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```
 
## References
<i id=1></i>[1] Ramit Sawhney, Shivam Agarwal, Atula Tejaswi Neerkaje, Nikolaos Aletras, Preslav Nakov, and Lucie Flek. 2022. Towards Suicide Ideation Detection Through Online Conversational Context. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 1716–1727. https://doi.org/10.1145/3477495.3532068.

<i id=2></i>[2] Sawhney, Ramit, Prachi Manchanda, Raj Singh, and Swati Aggarwal. "A computational approach to feature extraction for identification of suicidal ideation in tweets." In Proceedings of ACL 2018, Student Research Workshop, pp. 91-98. 2018.
