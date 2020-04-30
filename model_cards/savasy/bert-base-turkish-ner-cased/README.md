Thanks to
Stefan Schweter @stefan-it

I applied the following code to the datasets

```
for file in train.txt dev.txt test.txt labels.txt
do
  wget https://schweter.eu/storage/turkish-bert-wikiann/$file
done

cd ..
It will download the pre-processed datasets with training, dev and test splits and put them in a tr-data folder.

Run pre-training
After downloading the dataset, pre-training can be started. Just set the following environment variables:

export MAX_LENGTH=128
export BERT_MODEL=dbmdz/bert-base-turkish-128k-cased 
export OUTPUT_DIR=tr-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=625
export SEED=1

Then run pre-training:

python3 run_ner.py --data_dir ./tr-data \
--model_type bert \
--labels ./tr-data/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR-$SEED \
--max_seq_length $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict \
--fp16
```
