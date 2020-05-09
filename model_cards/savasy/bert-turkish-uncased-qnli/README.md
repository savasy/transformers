
# Turkish Bert-base QNLI Model

I fine-tuned Turkish-Bert-Model for Question-Answering problem with Turkish version of SQuAD; TQuAD 
https://huggingface.co/dbmdz/bert-base-turkish-uncased

# Data: TQuAD
I used following TQuAD data set which is Turkish Version of SQuAD

https://github.com/TQuad/turkish-nlp-qa-dataset

I convert the dataset into transformers glue data format of QNLI by the following script
SQuAD -> QNLI

```
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys

ff="dev-v0.1.json"
ff="train-v0.1.json"
dataset=json.load(open(ff))

i=0
for article in dataset['data']:
 title= article['title']
 for p in article['paragraphs']:
  context= p['context']
  for qa in p['qas']:
   answer= qa['answers'][0]['text']
   all_other_answers= list(set([e['answers'][0]['text'] for e in p['qas']]))
   all_other_answers.remove(answer)
   i=i+1
   print(i,qa['question'].replace(";",":") , answer.replace(";",":"),"entailment", sep="\t")
   for other in all_other_answers:
    i=i+1
    print(i,qa['question'].replace(";",":") , other.replace(";",":"),"not_entailment" ,sep="\t")
  
```


Under QNLI folder there are dev and test test
Training data looks like 
> 613     II.Friedrich’in bilginler arasındaki en önemli şahsiyet olarak belirttiği kişi kimdir?  filozof, kimyacı, astrolog ve çevirmen  not_entailment

> 614     II.Friedrich’in bilginler arasındaki en önemli şahsiyet olarak belirttiği kişi kimdir?  kişisel eğilimi ve özel temaslar nedeniyle      not_entailment

> 615     Michael Scotus’un mesleği nedir?        filozof, kimyacı, astrolog ve çevirmen  entailment

> 616     Michael Scotus’un mesleği nedir?        Palermo’ya      not_entailment





# Training

Training the model with following environment
```
export GLUE_DIR=./glue/glue_dataTR/QNLI
export TASK_NAME=QNLI
```

```
python3 run_glue.py \
  --model_type bert \
  --model_name_or_path dbmdz/bert-base-turkish-uncased\
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME/

```


# Evaluation Results


| Metric | Score |
|--- | ---|
| acc | 0.9124 |
| loss| 0.215 |




> See all my model
> https://huggingface.co/savasy





