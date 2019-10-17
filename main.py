#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import gc
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
import datetime
import time
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

path  = './Train/'

train_sales = pd.read_csv(path+'train_sales_data.csv')
train_search = pd.read_csv(path+'train_search_data.csv')
train_user = pd.read_csv(path+'train_user_reply_data.csv')

evaluation_public = pd.read_csv('evaluation_public.csv')
submit_example = pd.read_csv('submit_example.csv')


#数据处理
data = pd.concat([train_sales, evaluation_public], ignore_index=True)

data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth'])
data['label'] = data['salesVolume']
data.to_csv("1.csv")
data['id'] = data['id'].fillna(0).astype(int)             #fill the nan as '0'
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])#将待预测的数据条目的bodyType填上
                                                         #dataframe.duplicates('model')//remove the rows which repete in the column 'model'
                                                         #dataframe.set_index('model'), set the 'model' as index
                                                        #Series.map(self, arg, na_action=None)[source]: Map values of Series according to input correspondence.

for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
# map contents of bodyType to 0~3, the same as model: to 0~59
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']  #number of month from 2016/01



#提取统计特征
def get_stat_feature(df_):
    df = df_.copy()
    stat_feat = []
    df['model_adcode'] = df['adcode'] + df['model']        #different model OR province
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']  #唯一标识不同省份不同车型不同年月
    for col in tqdm(['label','popularity']):
        # shift
        for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i  #different model OR province OR month OR i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])
    for col in tqdm(['carCommentVolum','newsReplyVolum']):
        for i in [1,2,3,4,5,6]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col, i))
            df['model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'] + i  # different model OR province OR month OR i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col, i))
            df['shift_model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'].map(df_last[col])
    return df, stat_feat


def get_model_type(train_x,train_y,valid_x,valid_y,m_type='lgb'):
    if m_type == 'lgb':
        model = lgb.LGBMRegressor(
                                num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
                                max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2019,
                                n_estimators=2000, subsample=0.9, colsample_bytree=0.7,
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              categorical_feature=cate_feat,
              early_stopping_rounds=100, verbose=100)
    elif m_type == 'xgb':
        model = xgb.XGBRegressor(
                                max_depth=5 , learning_rate=0.05, n_estimators=2000,
                                objective='reg:gamma', tree_method = 'hist',subsample=0.9,
                                colsample_bytree=0.7, min_child_samples=5,eval_metric = 'rmse'
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              early_stopping_rounds=100, verbose=100)
    elif m_type == 'cat':
        model = CatBoostRegressor(
                                 depth=10, learning_rate=0.01, n_estimators=675, 
                                 reg_lambda=0.25, loss_function='RMSE'
                                  )
        model.fit(train_x, train_y,
                  cat_features=cate_feat,
                  eval_set=(valid_x, valid_y),plot=True)
    return model

def get_train_model(df_, m, m_type='lgb'):
    df = df_.copy()
    # 数据集划分
    st = 13
    all_idx   = (df['mt'].between(st , m-1))
    train_idx = (df['mt'].between(st , m-5))
    valid_idx = (df['mt'].between(m-4, m-4))
    test_idx  = (df['mt'].between(m  , m  ))
    print('all_idx  :',st ,m-1)
    print('train_idx:',st ,m-5)
    print('valid_idx:',m-4,m-4)
    print('test_idx :',m  ,m  )
    # 最终确认
    train_x = df[train_idx][features]
    train_y = df[train_idx]['label']
    valid_x = df[valid_idx][features]
    valid_y = df[valid_idx]['label']
    # get model
    model = get_model_type(train_x,train_y,valid_x,valid_y,m_type)
    # offline
    df['pred_label'] = model.predict(df[features])
    best_score = score(df[valid_idx])
    print("offline score",best_score)
    # online
    if m_type == 'lgb':
        model.n_estimators = model.best_iteration_ + 100
        model.fit(df[all_idx][features], df[all_idx]['label'], categorical_feature=cate_feat)
    elif m_type == 'xgb':
        model.n_estimators = model.best_iteration + 100
        model.fit(df[all_idx][features], df[all_idx]['label'])
    elif m_type == 'cat':
        model.fit(df[all_idx][features], df[all_idx]['label'],cat_features=cate_feat)#??
    
    df['forecastVolum'] = model.predict(df[features])
    print('valid mean:',df[valid_idx]['pred_label'].mean())
    print('true  mean:',df[valid_idx]['label'].mean())
    print('test  mean:',df[test_idx]['forecastVolum'].mean())
    # 阶段结果
    sub = df[test_idx][['id']]
    print(sub.shape)
    sub['forecastVolum'] = df[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    print(sub.shape)
    print(sub.columns)
    return sub,df[valid_idx]['pred_label']


def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2])
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)

df,stat_feat = get_stat_feature(data)

for month in [25, 26, 27, 28]:
    m_type = 'cat'

    data_df, stat_feat = get_stat_feature(data)

    num_feat = ['regYear'] + stat_feat
    cate_feat = ['adcode', 'bodyType', 'model', 'regMonth']

    if m_type == 'lgb':
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')

        print(data_df[i].dtype)
    elif m_type == 'xgb':
        lbl = LabelEncoder()  #sklearn.preprocessing.LabelEncoder: Encode labels with value between 0 and n_classes-1.
        for i in tqdm(cate_feat):
            data_df[i] = lbl.fit_transform(data_df[i].astype(str))
            print(data_df[i].dtype)
    features = num_feat + cate_feat

    sub, val_pred = get_train_model(data_df, month, m_type)
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values
sub = data.loc[(data.regMonth >= 1) & (data.regYear == 2018), ['id', 'salesVolume']]
sub.columns = ['id', 'forecastVolum']
sub[['id', 'forecastVolum']].round().astype(int).to_csv('CCF_sales.csv', index=False)

