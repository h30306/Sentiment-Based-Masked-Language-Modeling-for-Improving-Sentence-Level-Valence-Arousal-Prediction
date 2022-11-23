fold=3
device=0

CUDA_VISIBLE_DEVICES=$device python3.6 ./VA_BERT_no_mask/model/run_lm_finetuning.py \
            --output_dir=./VA_BERT_no_mask/output/original/fold"$fold" \
            --train_data_file=./data/train"$fold".csv \
            --eval_data_file=./data/val"$fold".csv \
            --test_data_file=./data/test"$fold".csv \
            --model_type=bert \
            --model_name_or_path=bert-base-chinese \
            --do_train --do_eval \
            --per_gpu_train_batch_size=256 \
            --per_gpu_eval_batch_size=256 \
            --overwrite_cache \
            --overwrite_output_dir \
            --add_tokens_data=./data/wikitext-2/DVA_dataset.csv \
            --num_train_epochs=500 \
            --save_steps=7 \
            --mlm \
            --fold=0

