#coding='utf-8'

__author__ = "Howard W. Chung"

from keras.layers import Embedding, Dense, LSTM, SimpleRNN, Bidirectional, GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import r2_score
from math import sqrt
import pandas as pd
import numpy as np
import pickle
import json
import os
import argparse


def split_text_from_column(df):
    return [([w.split('(')[:-1][0] for w in i.split(' ')]) for i in df[1]]

def read_data(path):
    df = pd.read_csv(path, header=None)
    return df
    
def build_input_data(args, tokenizer, max_len, evaluate=False, test=False):
    file_path = args.eval_data_file if evaluate else args.test_data_file if test else args.train_data_file
    df = read_data(file_path)
    df_split = split_text_from_column(df)
    df_tk = tokenizer.texts_to_sequences(df_split)
    df_data = pad_sequences(df_tk, maxlen=max_len, padding='post')
    return df_data

def build_target_data(args, target_name, evaluate=False, test=False):
    file_path = args.eval_data_file if evaluate else args.test_data_file if test else args.train_data_file
    df = read_data(file_path)
    column = 3 if (target_name == 'valence') else 4
    return list(df[column])

def build_tokenizer(token_data_path):
    tk_df = read_data(token_data_path)
    tk_split = split_text_from_column(tk_df)
    max_len = len(max(tk_split, key=len))
    tk = Tokenizer()
    tk.fit_on_texts(tk_split)
    return tk, max_len

def eval(predict, real):
    n = len(real)
    mae = sum(np.abs(real - predict))/n
    mse = sum(np.square(real - predict))/n
    rmse = sqrt(sum(np.square(real - predict))/n)
    r = r2_score(real, predict)
    return mae, rmse, mse, r

def Bidirectinal_Model(args, max_len, core_model_type, tk):
    model = Sequential()
    model.add(Embedding(len(tk.word_index)+1,  args.Embedding_size,  input_length=max_len))
    model.add(Bidirectional(core_model_type(args.hidden_size,
                                activation="relu",
                                return_sequences=True)))
    model.add(Bidirectional(core_model_type(args.hidden_size,
                                activation="relu")))
    model.add(Dense(1))
    return model

def RNN_Model(args, max_len, tk):
    model = Sequential()
    model.add(Embedding(len(tk.word_index)+1,  args.Embedding_size,  input_length=max_len))
    model.add(SimpleRNN(args.hidden_size, 
                activation='relu'))
    model.add(Dense(1))
    return model

def LSTM_Model(args, max_len, tk):
    model = Sequential()
    model.add(Embedding(len(tk.word_index)+1,  args.Embedding_size,  input_length=max_len))
    model.add(LSTM(args.hidden_size, 
                activation='relu'))
    model.add(Dense(1))
    return model

def GRU_Model(args, max_len, tk):
    model = Sequential()
    model.add(Embedding(len(tk.word_index)+1,  args.Embedding_size,  input_length=max_len))
    model.add(GRU(args.hidden_size, 
                activation='relu'))
    model.add(Dense(1))
    return model

def train(args, target_name, tokenizer, max_len, model):
    #Building Training Data
    print("Building Training Data...")
    x_train = build_input_data(args, tokenizer, max_len)
    y_train = build_target_data(args, target_name)
    #Building Validation Data
    print("Building Validation Data...")
    x_val = build_input_data(args, tokenizer, max_len, evaluate=True)
    y_val = build_target_data(args, target_name, evaluate=True)
    #Setting Model
    opt = Adam(learning_rate=args.learning_rate)
    model.compile(loss='mse', optimizer=opt)
    checkpointer = ModelCheckpoint(filepath="./{}/output/{}/{}/fold{}/{}_weights.hdf5".format(args.output_dir, args.bidirectional, args.model_type, args.fold, target_name), save_best_only=True)
    history = model.fit(x_train, y_train, validation_data=[x_val,y_val], callbacks=[checkpointer], epochs=args.epochs)
    history_df = pd.DataFrame(history.history) 
    history_df.to_csv("./{}/output/{}/{}/fold{}/{}_loss_df.csv".format(args.output_dir, args.bidirectional, args.model_type, args.fold, target_name), index=False)
    return model

def evaluate(args, target_name, tokenizer, max_len, model, test=False):
    evaluation_result={}
    #Building Evaluation Data
    if test:
        print("Building Testing Data...")
        x_val = build_input_data(args, tokenizer, max_len, test=True)
        y_val = build_target_data(args, target_name, test=True)
    else:
        print("Building Validation Data...")
        x_val = build_input_data(args, tokenizer, max_len, evaluate=True)
        y_val = build_target_data(args, target_name, evaluate=True)		
    predict, real = np.array(model.predict(x_val)).ravel(), np.array(y_val).ravel()
    mae, rmse, mse, r = eval(predict, real)
    predict_df = pd.DataFrame({'label':real, 'predict':predict})
    if target_name == 'valence':
        if test:
            predict_df.to_csv('./{}/output/{}/{}/fold{}/test_predict_V.csv'.format(args.output_dir, args.bidirectional, args.model_type, args.fold), encoding='utf-8', index=False)
        else:
            predict_df.to_csv('./{}/output/{}/{}/fold{}/val_predict_V.csv'.format(args.output_dir, args.bidirectional, args.model_type, args.fold), encoding='utf-8', index=False)
        evaluation_result.update({'mae_V':mae, 'rmse_V':rmse, 'mse_V':mse,'r_V':r})
    else:
        if test:
            predict_df.to_csv('./{}/output/{}/{}/fold{}/test_predict_A.csv'.format(args.output_dir, args.bidirectional, args.model_type, args.fold), encoding='utf-8', index=False)
        else:
            predict_df.to_csv('./{}/output/{}/{}/fold{}/val_predict_A.csv'.format(args.output_dir, args.bidirectional, args.model_type, args.fold), encoding='utf-8', index=False)
        evaluation_result.update({'mae_A':mae, 'rmse_A':rmse, 'mse_A':mse, 'r_A':r})
    return evaluation_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_file", 
        default="../data/train0.csv", 
        type=str, 
        help="The input training data file (a text file)."
    ) 
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default="../data/val0.csv",
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )

    parser.add_argument(
        "--test_data_file",
        default="../data/test.csv",
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )

    parser.add_argument(
        "--fold",
        default="0",
        type=str,
        help="which fold is trainning right now",
    )

    parser.add_argument(
        "--model_type",
        default="LSTM",
        type=str,
        help="The model name for trainning.",
    )

    parser.add_argument(
        "--add_tokens_data",
        type=str,
        default='./wikitext-2/DVA_dataset.csv',
        help='add tokens csv file_path',
    )

    parser.add_argument(
        "--target",
        type=str,
        default='valence',
        help='target column',
    )

    parser.add_argument(
        "--bidirectional",
        type=str,
        default='no_bidirectional',
        help='train on Bidirectional or not options:"bidirectional" or "no_bidirectional"',
    )

    parser.add_argument(
    	"--dimensional",
    	action="store_true",
    	help="do train on dimensional target",
    )
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--Embedding_size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run test on the test set.")
    args = parser.parse_args()

    create_path = './{}/output/{}/{}/fold{}/'.format(args.output_dir, args.bidirectional, args.model_type, args.fold)
    if not os.path.isdir(create_path):
        print("Creating Output Path")
        os.makedirs(create_path)

    tokenizer_path = "./{}/output/tokenizer.pickle".format(args.output_dir)
    if not os.path.isdir(tokenizer_path):
        #Build Tokenizer
        print("Building Tokenizer..")
        tokenizer, max_len = build_tokenizer(args.add_tokens_data)
        with open('./{}/output/tokenizer.pickle'.format(args.output_dir), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./{}/output/max_len.text'.format(args.output_dir), 'w') as f:
            f.write(str(max_len))
            f.close()
    else:
        #Load Tokenizer
        print("Loading Tokenizer..")
        with open('./{}/output/tokenizer.pickle'.format(args.output_dir), 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('./{}/output/max_len.text'.format(args.output_dir), 'r') as f:
            max_len = int(f.readline())

    MODEL_CLASSES = {
        "RNN":RNN_Model(args, max_len, tokenizer),
        "LSTM":LSTM_Model(args, max_len, tokenizer),
        "GRU":GRU_Model(args, max_len, tokenizer),
    }
    CORE_MODEL_CLASSES = {
        "RNN":SimpleRNN,
        "LSTM":LSTM,
        "GRU":GRU,
    }

    if args.do_train:
        if args.bidirectional == 'bidirectional':
            model = Bidirectinal_Model(args, max_len, CORE_MODEL_CLASSES[args.model_type], tokenizer)
        else:
            model = MODEL_CLASSES[args.model_type]
        model.summary()
        if args.dimensional:
            model_V = train(args, 'valence', tokenizer, max_len, model)
            model_A = train(args, 'arousal', tokenizer, max_len, model)
            val_result = evaluate(args, 'valence', tokenizer, max_len, model_V)
            val_result_A = evaluate(args, 'arousal', tokenizer, max_len, model_A)
            val_result.update(val_result_A)
        else:
            model = train(args, args.target, tokenizer, max_len, model)
            val_result = evaluate(args, args.target, tokenizer, max_len, model)

        with open('./{}/output/{}/{}/fold{}/val_evaluation_result.json'.format(args.output_dir, args.bidirectional, args.model_type, args.fold), 'w') as fp:
            json.dump(val_result, fp)

    #Do Evaluation only
    if args.do_eval:
        if args.dimensional:
            try:
                model_V = load_model(create_path+'valence_weights.hdf5')
                print("Load Model_V From Pretrained Weight")
                model_V.summary()
            except:
                raise ImportError("valence_weights.hdf5 model not found please set --do_train & --do_eval")
            try:
                model_A = load_model(create_path+'arousal_weights.hdf5')
                print("Load Model_A From Pretrained Weight")
                model_A.summary()
            except:
                raise ImportError("arousal_weights.hdf5 model not found please set --do_train & --do_eval")
            test_result = evaluate(args, 'valence', tokenizer, max_len, model_V, test=True)
            test_result_A = evaluate(args, 'arousal', tokenizer, max_len, model_A, test=True)
            test_result.update(test_result_A)
        else:
            try:
                model = load_model(create_path+'{}_weights.hdf5'.format(args.target))
                print("Load Model From Pretrained Weight")
                model.summary()
            except:
                raise ImportError("{}_weights.hdf5 model not found please set --do_train & --do_eval".format(args.target))  
            test_result = evaluate(args, args.target, tokenizer, max_len, model, test=True)  
        with open('./{}/output/{}/{}/fold{}/test_evaluation_result.json'.format(args.output_dir, args.bidirectional, args.model_type, args.fold), 'w') as fp:
            json.dump(test_result, fp)  

if __name__ == "__main__":
    main()
