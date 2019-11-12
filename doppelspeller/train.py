#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import _pickle as pickle
import random
from collections import Counter
from itertools import product
from datetime import datetime

import unicodedata
import pandas as pd
import numpy as np
import xgboost as xgb
from datasketch import MinHashLSHForest, MinHash
from fuzzywuzzy import fuzz

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


SEQUENCES_OF = 3
NUM_PERM = 128


# In[3]:


get_ipython().run_cell_magic('time', '', '\n%run common.ipynb')


# In[4]:


in_top = 10
with open('similar_programs_{}.dump'.format(in_top), 'rb') as fl:
    lsh_forest_data = pickle.load(fl)
training_data_negative = lsh_forest_data.pop(-1)


# In[5]:


ground_truth_mapping = ground_truth.set_index('company_id').copy(deep=True)
ground_truth_mapping = ground_truth_mapping.to_dict()['transformed_name']


# In[6]:


train_data = pd.read_csv('STrain.csv', delimiter='|')
train_data.loc[:, 'transformed_name'] = train_data.loc[:, 'name'].apply(lambda x: convert_text(x))
train_data = train_data.set_index('company_id')
del train_data['name']
train_data_mapping = train_data.to_dict()['transformed_name']

train_data_negatives_mapping = train_data[train_data.index == -1].copy(deep=True)
train_data_negatives_mapping = train_data_negatives_mapping.set_index('train_index').to_dict()['transformed_name']


# In[7]:


generated_training_data = pd.read_pickle('generated_training_data.dump')


# In[8]:


generated_training_data.head()


# In[9]:


get_ipython().run_cell_magic('time', '', "\ntraining_rows_generated = []\nfor train_index, row in generated_training_data.iterrows():\n    truth_company_name = row['transformed_name']\n    company_name = row['generated_misspelled']\n    training_rows_generated.append(\n        [1, company_name, company_name.split(' '), truth_company_name, truth_company_name.split(' '), 1])")


# In[10]:


get_ipython().run_cell_magic('time', '', "\ntraining_rows_negative = []\nfor train_index, companies in training_data_negative.items():\n    company_name = train_data_negatives_mapping[train_index]\n    for truth_company_id in companies:\n        truth_company_name = ground_truth_mapping[truth_company_id]\n        truth_company_name_words = truth_company_name.split(' ')\n        company_name_words = company_name.split(' ')\n        training_rows_negative.append(\n            [2, company_name, company_name_words, truth_company_name, truth_company_name_words, 0])")


# In[11]:


get_ipython().run_cell_magic('time', '', "\ntraining_rows = []\nfor company_id, companies in lsh_forest_data.items():\n    company_name = train_data_mapping[company_id]\n    for truth_company_id in companies:\n        truth_company_name = ground_truth_mapping[truth_company_id]\n        truth_company_name_words = truth_company_name.split(' ')\n        company_name_words = company_name.split(' ')\n        training_rows.append(\n            [3, company_name, company_name_words, truth_company_name, truth_company_name_words, int(company_id == truth_company_id)])")


# In[12]:


training_rows_final = training_rows_negative + training_rows + training_rows_generated


# In[13]:


train = pd.DataFrame(training_rows_final, columns=[
    'kind', 'company_name', 'company_name_words', 'truth_company_name', 'truth_company_name_words', 'target'])


# In[14]:


get_ipython().run_cell_magic('time', '', "\ntrain.loc[:, 'number_of_characters'] = train.loc[:, 'company_name'].apply(\n    lambda x: len(x)\n)\ntrain.loc[:, 'truth_number_of_characters'] = train.loc[:, 'truth_company_name'].apply(\n    lambda x: len(x)\n)\ntrain.loc[:, 'number_of_words'] = train.loc[:, 'company_name_words'].apply(\n    lambda x: len(x)\n)\ntrain.loc[:, 'truth_number_of_words'] = train.loc[:, 'truth_company_name_words'].apply(\n    lambda x: len(x)\n)\ntrain.loc[:, 'levenshtein'] = list(\n    map(lambda x: fuzz.ratio(x[0], x[1]), zip(train.loc[:, 'company_name'], train.loc[:, 'truth_company_name'])))")


# In[15]:


train.head()


# In[16]:


len(train)


# In[17]:


# Takes ~9 minutes

st = time.time()

extra_features = list(map(lambda x: construct_features(x[0], x[1], x[2]), zip(
    train.loc[:, 'truth_number_of_words'], 
    train.loc[:, 'truth_company_name_words'], 
    train.loc[:, 'company_name'])))

print(round((time.time() - st)/60, 2))


# In[19]:


columns = ['truth_{}th_word_length'.format(x+1) for x in range(15)]
columns += ['truth_{}th_word_probability'.format(x+1) for x in range(15)]
columns += ['truth_{}th_word_probability_rank'.format(x+1) for x in range(15)]
columns += ['truth_{}th_word_best_match_score'.format(x+1) for x in range(15)]
columns.append('reconstructed_score')


# In[20]:


extra = pd.DataFrame(extra_features, columns=columns)


# In[21]:


train = train.merge(extra, right_index=True, left_index=True)


# In[22]:


evaluation_generated = train.loc[train['kind'] == 1, :].sample(frac=0.05).copy(deep=True)
evalutaion_negative = train.loc[train['kind'] == 2, :].sample(frac=0.1).copy(deep=True)
evaluation_positive = train.loc[train['kind'] == 3, :].sample(frac=0.05).copy(deep=True)


# In[23]:


evaluation = pd.concat([evaluation_generated, evalutaion_negative, evaluation_positive])


# In[24]:


evaluation_indexes = list(evaluation_generated.index) + list(evalutaion_negative.index) + list(evaluation_positive.index)


# In[25]:


train = train.loc[~train.index.isin(evaluation_indexes), :].copy(deep=True).reset_index(drop=True)


# In[26]:


evaluation = evaluation.reset_index(drop=True)


# In[27]:


del train['kind']
del evaluation['kind']


# In[28]:


len(train), len(evaluation)


# In[29]:


# train.to_pickle('train.dump')
# evaluation.to_pickle('evaluation.dump')


# In[30]:


# train = pd.read_pickle('train.dump')
# evaluation = pd.read_pickle('evaluation.dump')


# In[31]:


train_set = train.loc[:, ~train.columns.isin(
    ['company_name', 'company_name_words', 'truth_company_name', 'truth_company_name_words', 'target'])]
train_set_target = train.loc[:, 'target']

d_train = xgb.DMatrix(train_set.values, label=train_set_target, feature_names=train_set.columns)

evaluation_set = evaluation.loc[:, ~evaluation.columns.isin(
    ['company_name', 'company_name_words', 'truth_company_name', 'truth_company_name_words', 'target'])]
evaluation_set_target = evaluation.loc[:, 'target']

d_evaluation = xgb.DMatrix(evaluation_set.values, label=evaluation_set_target, feature_names=evaluation_set.columns)


# In[32]:


num_rows_train = len(train_set)


# In[33]:


def custom_error(predictions, train_or_eval):
    actuals = train_or_eval.get_label()
    is_train_set = False
    if len(actuals) == num_rows_train:
        is_train_set = True
        
    threshold = 0.5
    
    predictions_negative_indexes = (predictions < threshold).nonzero()[0]
    predictions_positive_indexes = (predictions >= threshold).nonzero()[0]
    
    false_negative_cost = sum(actuals[predictions_negative_indexes])
    false_positive_cost = sum(actuals[predictions_positive_indexes] == 0) * 5
    
    cost = false_negative_cost + false_positive_cost

    return 'custom-error', cost


# In[34]:


def weighted_logloss(predictions, train):
    beta = 5
    actuals = train.get_label()
    gradient = predictions * (beta + actuals - beta * actuals) - actuals
    hessian = predictions * (1 - predictions) * (beta + actuals - beta * actuals)
    return gradient, hessian


# In[35]:


scale_pos_weight = sum(train_set_target == 0) / sum(train_set_target == 1)
print(scale_pos_weight)


# In[37]:


get_ipython().run_cell_magic('time', '', "\nwatch_list = [(d_train, 'train'), (d_evaluation, 'evaluation')]\nparams = {\n    'params': {\n        'max_depth': 5,\n        'eta': 0.1,\n        'nthread': 4,\n        'min_child_weight': 1,\n        'eval_metric': 'auc',\n        'objective': 'reg:logistic',\n        'scale_pos_weight': scale_pos_weight,\n        'subsample': 1,\n    },\n    'num_boost_round': 1000,\n    'verbose_eval': True,\n    'early_stopping_rounds': 50,\n}\n\nmodel = xgb.train(\n    dtrain=d_train,\n    evals=watch_list,\n    feval=custom_error,\n    obj=weighted_logloss,\n    maximize=False,\n    **params\n)")


# In[ ]:


def get_xgb_feats_importance(reg):
    fscore = reg.get_fscore()

    feat_importances = []
    for ft, score in fscore.items():
        feat_importances.append({'feature': ft, 'importance': score})

    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(by='importance', ascending=False).reset_index(drop=True)
    feat_importances['importance'] /= feat_importances['importance'].sum()
    return feat_importances


# In[ ]:


features_importance = get_xgb_feats_importance(model)


# In[ ]:


ax = features_importance.head(20).plot.barh(x='feature', y='importance')


# In[ ]:


sum(features_importance.head(20)['importance'])


# In[ ]:


postfix = str(datetime.now()).replace(' ', '').replace(':', '-').replace('.', '-')
with open('model-{}.dump'.format(postfix), 'wb') as fl:
    pickle.dump(model, fl)


# In[ ]:




