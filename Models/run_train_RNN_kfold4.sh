fold=4
device=1
model="RNN LSTM GRU"

for m in $model;
	do
	CUDA_VISIBLE_DEVICES=$device python3.6 ./VA_RNN/model/RNN_text_classification.py \
				--output_dir=VA_RNN \
				--train_data_file=./data/train"$fold".csv \
				--eval_data_file=./data/val"$fold".csv \
				--test_data_file=./data/test"$fold".csv \
				--fold=$fold \
				--model_type=$m \
				--add_tokens_data=./data/wikitext-2/DVA_dataset.csv \
				--bidirectional=bidirectional \
				--dimensional \
				--epochs=500 \
				--do_train \
				--do_eval
	done
