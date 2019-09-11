# A Pytorch Implementations of BERT-based Relation Classification

This is a stable pytorch implementation of ``Enriching Pre-trained Language Model with Entity Information for Relation Classification`` https://arxiv.org/abs/1905.08284.
### Requirements:
 
Python version >= 3.6 (recommended)
Pytorch version >= 1.1 (recommended)
pytorch-transformers: https://github.com/huggingface/pytorch-transformers


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
09/11/2019 16:36:33 - INFO - __main__ -   Loading features from cached file ./dataset/cached_dev_bert-base-uncased_128_semeval
09/11/2019 16:36:33 - INFO - __main__ -   Saving features into cached file ./dataset/cached_dev_bert-base-uncased_128_semeval
09/11/2019 16:36:34 - INFO - __main__ -   ***** Running evaluation  *****
09/11/2019 16:36:34 - INFO - __main__ -     Num examples = 2717
09/11/2019 16:36:34 - INFO - __main__ -     Batch size = 8
Evaluating: 100%|████████████████████████████████████████████████████| 340/340 [00:46<00:00,  7.24it/s]
09/11/2019 16:37:21 - INFO - __main__ -   ***** Eval results  *****
09/11/2019 16:37:21 - INFO - __main__ -     acc = 0.8502024291497976
09/11/2019 16:37:21 - INFO - __main__ -     acc_and_f1 = 0.8502024291497976
09/11/2019 16:37:21 - INFO - __main__ -     f1 = 0.8502024291497976
```

3. Evaluate using the official script for SemEval task-8

```
> cd eval
> bash test.sh
> cat res.txt
```

```
(the reported result in the paper ) MACRO-averaged result (excluding Other): 89.25
(this pytorch implementation) MACRO-averaged result (excluding Other): 88.75
```


