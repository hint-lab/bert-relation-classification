# A Pytorch Implementation of BERT-based Relation Classification

This is a stable pytorch implementation of ``Enriching Pre-trained Language Model with Entity Information for Relation Classification`` https://arxiv.org/abs/1905.08284.
### Requirements:
 
Python version >= 3.6 (recommended)

Pytorch version >= 1.1 (recommended)

pytorch-transformers: https://github.com/huggingface/pytorch-transformers  
!!! pytorch-transformers = 1.1 




### Tutorial of the code

1. Download the project and prepare the data

```
> git clone https://github.com/wang-h/bert-relation-classification
> cd bert-relation-classification 
```

2. Train the bert-based classification model

```
> python bert.py --config config.ini
```

```
...
09/11/2019 16:36:31 - INFO - pytorch_transformers.modeling_utils -   loading weights file /tmp/semeval/pytorch_model.bin
09/11/2019 16:36:33 - INFO - __main__ -   Loading features from cached file ./dataset/cached_dev_bert-large-uncased_128_semeval
09/11/2019 16:36:33 - INFO - __main__ -   Saving features into cached file ./dataset/cached_dev_bert-large-uncased_128_semeval
09/11/2019 16:36:34 - INFO - __main__ -   ***** Running evaluation  *****
09/11/2019 16:36:34 - INFO - __main__ -     Num examples = 2717
09/11/2019 16:36:34 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|████████████████████████████████████████████████████| 340/340 [00:46<00:00,  7.24it/s]
09/11/2019 16:37:21 - INFO - __main__ -   ***** Eval results  *****  
10/07/2019 10:02:23 - INFO - __main__ -     acc = 0.8579315421420685
10/07/2019 10:02:23 - INFO - __main__ -     acc_and_f1 = 0.8579315421420685
10/07/2019 10:02:23 - INFO - __main__ -     f1 = 0.8579315421420685
```

3. Evaluate using the official script for SemEval task-8

```
> cd eval
> bash test.sh
> cat res.txt
```

```
(the result reported in the paper, tensorflow) MACRO-averaged result (excluding Other, uncased-large-model): 89.25 
(this pytorch implementation) MACRO-averaged result (excluding Other, uncased-large-model): 89.25 (same)
```

I also have the source code written in tensorflow. Feel free to contact me if you need it.

We also appreciate if you could cite our recent paper with the best result (90.36).

Enhancing Relation Extraction Using Syntactic Indicators and Sentential Contexts

https://arxiv.org/abs/1912.01858

or 
  
@article{10.1145/3402885,
author = {Wang, Hao and Tao, Qiongxing and Du, Siyuan and Luo, Xiangfeng},
title = {An Extensible Framework of Leveraging Syntactic Skeleton for Semantic Relation Classification},
year = {2020},
issue_date = {September 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {19},
number = {6},
issn = {2375-4699},
url = {https://doi.org/10.1145/3402885},
doi = {10.1145/3402885}, 
journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.}, 
articleno = {77},
numpages = {21} 
}
  
