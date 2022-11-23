import os
import numpy as np
import re
import json
import pandas as pd
from operator import add
from IPython import embed
from sklearn.metrics import median_absolute_error
os.listdir()

model_type = ['VA_BERT_random_mask', 
              'VA_BERT_mask_sentiment', 
              'VA_BERT_mask_sentiment_prob_V', 
              'VA_BERT_mask_sentiment_prob_reverse_V', 
              'VA_BERT_mask_sentiment_prob_A', 
              'VA_BERT_mask_sentiment_prob_reverse_A',
              'VA_BERT_mask_sentiment_prob_Union', 
              'VA_BERT_mask_sentiment_prob_Intersection',
              'VA_BERT_mask_sentiment_prob_paper', 
              'VA_BERT_mask_sentiment_prob_paper_multilabel',
              'VA_BERT_mask_sentiment_prob_Union_multilabel',
              'VA_BERT_mask_sentiment_prob_Intersection_multilabel',
              'VA_BERT_no_mask']
fold = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']
mlmp = ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', 'best', 'average']
data_type = ['test_evaluation_result', 'val_evaluation_result']
evaluation_type = ['mae_V', 'mae_A', 'rmse_V', 'rmse_A', 'r_V', 'r_A', 'mape_V', 'mape_A', 'mdae_V', 'mdae_A', 'Average_VA', 'rank_r_V', 'rank_r_A', 'rank_mae_V', 'rank_mae_A', 'rank_rmse_V', 'rank_rmse_A', 'rank_mape_V', 'rank_mape_A', 'rank_mdae_V', 'rank_mdae_A','average_rank']
alpha_list = ['0.6', '0.8', '1']

val_result_V = pd.DataFrame(index=mlmp, columns=model_type).fillna(float(0))
test_result_V = pd.DataFrame(index=mlmp, columns=model_type).fillna(float(0))
val_result_A = pd.DataFrame(index=mlmp, columns=model_type).fillna(float(0))
test_result_A = pd.DataFrame(index=mlmp, columns=model_type).fillna(float(0))
test_compare_table = pd.DataFrame(index=evaluation_type, columns=model_type).fillna(float(0))
val_compare_table = pd.DataFrame(index=evaluation_type, columns=model_type).fillna(float(0))
val_result_alpha = pd.DataFrame(index=alpha_list, columns=model_type).fillna(float(0))
test_result_alpha = pd.DataFrame(index=alpha_list, columns=model_type).fillna(float(0))

def evaluate(et, df, t):
    try:
        if t=='V':
            try:
                y_predict = np.array([eval(i) for i in list(df['predictV'])]).ravel()
            except:
                y_predict = list(df['predictV'])
            y_true = np.array([eval(i) for i in list(df['labelV'])]).ravel()
        else:
            try:
                y_predict = np.array([eval(i) for i in list(df['predictA'])]).ravel()
            except:
                y_predict = list(df['predictA'])
            y_true = np.array([eval(i) for i in list(df['labelA'])]).ravel()
        if et[:4] == 'mape':
            return median_absolute_error(y_true, y_predict)
        else:
            return np.mean(np.abs((y_true - y_predict) / y_true)) * 100
    except:
        embed()

def rank(table, e_type, m_type):
    e = list(table.loc[e_type])
    e.sort()
    d=dict()
    if e_type[:2] == 'r_':
        e.reverse()
    for i,v in enumerate(e):
        d[v]=i+1
    for mt in m_type:
        index_name = 'rank_'+e_type
        table[mt][index_name] = d[table[mt][e_type]]
    #embed()
    return table

#抓每個MLM prob 的 mae_V, mae_A
for mt in model_type:
    if mt != 'VA_BERT_no_mask':
        for f in fold:
            for m in mlmp[:-2]:
                for dt in data_type:
                    f_ = open('./{}/output/{}/output{}/alpha{}/{}{}.json'.format(mt, f, m, alpha, dt, m), 'r', encoding='utf-8')
                    d = json.load(f_)
                    if dt == 'test_evaluation_result':
                        test_result_V[mt][m] += d['mae_V']
                        test_result_A[mt][m] += d['mae_A']
                    else:
                        val_result_V[mt][m] += d['mae_V']
                        val_result_A[mt][m] += d['mae_A']
                    for a in alpha_list:
                        f_ = open('./{}/output/{}/output{}/alpha{}/{}{}.json'.format(mt, f, m, a, dt, m), 'r', encoding='utf-8')
                        d = json.load(f_)
                        if dt == 'test_evaluation_result':
                            val_result_alpha[mt][a] += d['rmse_V']
                            test_result_alpha[mt][a] += d['rmse_V']
                        else:
                            val_result_alpha[mt][a] += d['rmse_V']
                            test_result_alpha[mt][a] += d['rmse_V']

    else:
        test_original_V = 0
        test_original_A = 0
        val_original_V = 0
        val_original_A = 0
        for f in fold:
            for dt in data_type:
                f_ = open('./{}/output/original/{}/{}.json'.format(mt, f, dt), 'r', encoding='utf-8')
                d = json.load(f_)
                if dt == 'test_evaluation_result':
                    test_original_V += d['mae_V']
                    test_original_A += d['mae_A']
                else:
                    val_original_V += d['mae_V']
                    val_original_A += d['mae_A']
        print('test_mae_V', test_original_V/5)
        print('test_mae_A', test_original_A/5)
        print('val_mae_V', val_original_V/5)
        print('val_mae_A', val_original_A/5)

val_result_alpha = val_result_alpha.div(5)
test_result_alpha = test_result_alpha.div(5)
val_result_V = val_result_V.div(5)
test_result_V = test_result_V.div(5)
val_result_A = val_result_A.div(5)
test_result_A = test_result_A.div(5)

#embed()

for mt in model_type:
    best = min(val_result_V[mt][:-2])
    val_result_V[mt]['best'] = best
    val_result_V[mt]['average'] = sum(val_result_V[mt][:-2])/19

    best = min(test_result_V[mt][:-2])
    test_result_V[mt]['best'] = best
    test_result_V[mt]['average'] = sum(test_result_V[mt][:-2])/19

    best = min(val_result_A[mt][:-2])
    val_result_A[mt]['best'] = best
    val_result_A[mt]['average'] = sum(val_result_A[mt][:-2])/19

    best = min(test_result_A[mt][:-2])
    test_result_A[mt]['best'] = best
    test_result_A[mt]['average'] = sum(test_result_A[mt][:-2])/19

#抓其他的指標
for mt in model_type:
    for t in ['V','A']:
        if t == 'V':
            m = test_result_V[[mt]][:-2].idxmin()[0]
        else:
            m = test_result_A[[mt]][:-2].idxmin()[0]
        for f in fold:
            for dt in data_type:
                try:
                    f_ = open('./{}/output/{}/output{}/alpha{}/{}{}.json'.format(mt, f, m, alpha, dt, m), 'r', encoding='utf-8')
                except:
                    f_ = open('./{}/output/original/{}/{}.json'.format(mt, f, dt), 'r', encoding='utf-8')
                d = json.load(f_)
                for et in evaluation_type[:6]:
                    if dt == 'test_evaluation_result':
                        if (t == 'V') & (t in et):
                            test_compare_table[mt][et] += d[et]
                        elif (t == 'A') & (t in et):
                            test_compare_table[mt][et] += d[et]
                        else:
                            continue
                    else:
                        if (t == 'V') & (t in et):
                            val_compare_table[mt][et] += d[et]
                        elif (t == 'A') & (t in et):
                            val_compare_table[mt][et] += d[et]
                try:
                    f_ = pd.read_csv('./{}/output/{}/output{}/alpha{}/{}_predict{}.csv'.format(mt, f, m, alpha, dt[:4], m))
                except:
                    try:
                        f_ = pd.read_csv('./{}/output/{}/output{}/alpha{}/{}_predict.csv'.format(mt, f, m, alpha, dt[:4]))
                    except:
                        try:
                            f_ = pd.read_csv('./{}/output/{}/output{}/alpha{}/{}_predict.csv'.format(mt, f, m, alpha, dt[:3]))
                        except:
                            try:
                                f_ = pd.read_csv('./{}/output/{}/output{}/alpha{}/{}_predict{}.csv'.format(mt, f, m, alpha, dt[:3], m))
                            except:
                                try:
                                    f_ = pd.read_csv('./{}/output/original/{}/{}_predict.csv'.format(mt, f, dt[:3]))
                                except:
                                    f_ = pd.read_csv('./{}/output/original/{}/{}_predict.csv'.format(mt, f, dt[:4]))
                for et in evaluation_type[6:10]:
                        if dt == 'test_evaluation_result':
                            if (t == 'V') & (t in et):
                                test_compare_table[mt][et] += evaluate(et, f_, t)
                            elif (t == 'A') & (t in et):
                                test_compare_table[mt][et] += evaluate(et, f_, t)
                            else:
                                continue
                        else:
                            if (t == 'V') & (t in et):
                                val_compare_table[mt][et] += evaluate(et, f_, t)
                            elif (t == 'A') & (t in et):
                                val_compare_table[mt][et] += evaluate(et, f_, t)

        for table in [test_compare_table, val_compare_table]:
            table[mt]['Average_VA'] = (table[mt]['mae_V']+table[mt]['mae_A'])/2

test_compare_table = test_compare_table.div(5)
val_compare_table = val_compare_table.div(5)

#算排名
for e_type in ['mae_V', 'mae_A', 'r_V', 'r_A', 'rmse_V', 'rmse_A', 'mape_V', 'mape_A', 'mdae_V', 'mdae_A']:
    test_compare_table = rank(test_compare_table, e_type, model_type)
    val_compare_table = rank(val_compare_table, e_type, model_type)

for mt in model_type: 
    for table in [test_compare_table, val_compare_table]:
        table[mt]['average_rank'] = (table[mt]['rank_mae_V']+table[mt]['rank_mae_A'])/2


val_result_V.to_csv('val_result_V.csv', encoding='utf-8')
test_result_V.to_csv('test_result_V.csv', encoding='utf-8')
val_result_A.to_csv('val_result_A.csv', encoding='utf-8')
test_result_A.to_csv('test_result_A.csv', encoding='utf-8')
test_compare_table.to_csv('test_compare_table.csv', encoding='utf-8')
val_compare_table.to_csv('val_compare_table.csv', encoding='utf-8')
val_result_alpha.to_csv('val_result_alpha.csv', encoding='utf-8')
test_result_alpha.to_csv('test_result_alpha.csv', encoding='utf-8')