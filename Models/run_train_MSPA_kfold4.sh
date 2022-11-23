fold=4
device=0

for m in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5;
    do
    for alpha in 0.6 0.8 1;
        do
        echo ==================================================$m============================================================
        CUDA_VISIBLE_DEVICES=$device python3.6 ./VA_BERT_mask_sentiment_prob_A/model/run_lm_finetuning.py \
                                --train_data_file=./data/train"$fold".csv \
                                --eval_data_file=./data/val"$fold".csv \
                                --test_data_file=./data/test"$fold".csv \
                                --output_dir=./VA_BERT_mask_sentiment_prob_A/output/fold"$fold"/output"$m"/alpha"$alpha" \
                                --sentiment_w_path=./data/wikitext-2/mask_prob_A.json \
                                --model_type=bert \
                                --model_name_or_path=bert-base-chinese \
                                --do_train \
                                --do_eval \
                                --mlm \
                                --per_gpu_train_batch_size=256 \
                                --per_gpu_eval_batch_size=256 \
                                --overwrite_cache \
                                --overwrite_output_dir \
                                --mlm_probability=$m \
                                --add_tokens_data=./data/wikitext-2/DVA_dataset.csv \
                                --num_train_epochs=500 \
                                --save_steps=7 \
                                --fold=$fold \
                                --alpha=$alpha
        done
    done

python3.6 ../RunningDoneLine.py --device=$device