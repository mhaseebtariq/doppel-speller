#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import json
import _pickle as pickle
import random
import sqlite3
from collections import Counter
from itertools import product
from concurrent.futures import ProcessPoolExecutor

import unicodedata
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import xgboost as xgb
from datasketch import MinHashLSHForest, MinHash


# In[2]:


SEQUENCES_OF = 3
NUM_PERM = 128


# In[3]:


get_ipython().run_cell_magic('time', '', '\n%run common.ipynb')


# In[4]:


with open('lsh_forest.dump', 'rb') as fl:
    lsh_forest = pickle.load(fl)


# In[5]:


ground_truth_mapping = ground_truth.set_index('company_id').copy(deep=True)
ground_truth_mapping = ground_truth_mapping.to_dict()['transformed_name']
ground_truth_mapping = {
    k: {
        'truth_company_name': v, 
        'truth_company_name_words': v.split(' '),
        'truth_number_of_characters': len(v),
        'truth_number_of_words': len(v.split(' ')),
    } for k, v in ground_truth_mapping.items()
}
ground_truth_mapping_reversed = {v['truth_company_name']: k for k, v in ground_truth_mapping.items()}


# In[6]:


get_ipython().run_cell_magic('time', '', "\ntest_data = pd.read_csv('STest.csv', delimiter='|')\ntest_data.loc[:, 'transformed_name'] = test_data.loc[:, 'name'].apply(lambda x: convert_text(x))\ndel test_data['name']\ndel test_data['test_index']\ntest_data.loc[:, 'sequences'] = test_data.loc[:, 'transformed_name'].apply(\n    lambda x: get_sequences(x, SEQUENCES_OF)\n)")


# In[7]:


connection = sqlite3.connect('data.db')
cursor = connection.cursor()

# Drop table
cursor.execute('''DROP TABLE IF EXISTS predictions;''')

# Create table
cursor.execute('''
    CREATE TABLE predictions (test_id INTEGER, 
                              title_to_match TEXT, 
                              best_match TEXT, 
                              best_match_id INTEGER, 
                              best_match_probability FLOAT);
''')

# Save (commit) the changes
connection.commit()


# In[8]:


def get_nearest_matches(test_id):
    query = 'SELECT matches from neighbours WHERE test_id={}'.format(test_id)
    cursor.execute(query)
    return json.loads(cursor.fetchone()[0])


# In[9]:


def save_prediction(test_id, title_to_match, best_match, best_match_id, best_match_probability):
    query = f"INSERT INTO predictions VALUES ({test_id}, '{title_to_match}', '{best_match}', {best_match_id}, {best_match_probability});"
    cursor.execute(query)
    connection.commit()
    return


# In[10]:


get_ipython().run_cell_magic('time', '', "\n# 1st Step\n\nmatched_so_far = []\n\ntest_data.loc[:, 'exact'] = test_data.loc[:, 'transformed_name'].apply(\n    lambda x: ground_truth_mapping_reversed.get(x, False))\n\nfor index, row in test_data.loc[test_data['exact'] != False, :].iterrows():\n    title_to_match = row['transformed_name']\n    best_match_id = row['exact']\n    best_match = ground_truth_mapping[best_match_id]['truth_company_name']\n    \n    matched_so_far.append(index)\n    save_prediction(index, title_to_match, best_match, best_match_id, 1.0)")


# In[11]:


len(test_data.loc[~test_data.index.isin(matched_so_far), :])


# In[12]:


get_ipython().run_cell_magic('time', '', "\n# 2nd Step\n\ncount = 0\nfor index, row in test_data.loc[~test_data.index.isin(matched_so_far), :].iterrows():\n    count += 1\n    if not(count % 10000):\n        print(count)\n\n    title_to_match = row['transformed_name']\n    best_match_ids = get_nearest_matches(index)\n    \n    if not best_match_ids:\n        matched_so_far.append(index)\n        save_prediction(index, title_to_match, None, -1, 0.0)\n        continue\n    \n    matches = [ground_truth_mapping[best_match_id]['truth_company_name'] for best_match_id in best_match_ids]\n    ratios = [fuzz.ratio(best_match, title_to_match) for best_match in matches]\n    arg_max = np.argmax(ratios)\n    max_ratio = ratios[arg_max]\n    best_match_id = best_match_ids[arg_max]\n    if max_ratio > 94:\n        matched_so_far.append(index)\n        best_match = matches[arg_max]\n        save_prediction(index, title_to_match, best_match, best_match_id, 1.0)\n    else:\n        ratios = [fuzz.token_sort_ratio(best_match, title_to_match) for best_match in matches]\n        arg_max = np.argmax(ratios)\n        max_ratio = ratios[arg_max]\n        best_match_id = best_match_ids[arg_max]\n        if max_ratio > 94:\n            matched_so_far.append(index)\n            best_match = matches[arg_max]\n            save_prediction(index, title_to_match, best_match, best_match_id, 1.0)")


# In[13]:


len(test_data.loc[~test_data.index.isin(matched_so_far), :])


# In[14]:


with open('model.dump', 'rb') as fl:
    model = pickle.load(fl)


# In[15]:


feature_names = [
    'number_of_characters', 'truth_number_of_characters', 'number_of_words',
    'truth_number_of_words', 'levenshtein', 'truth_1th_word_length',
    'truth_2th_word_length', 'truth_3th_word_length',
    'truth_4th_word_length', 'truth_5th_word_length',
    'truth_6th_word_length', 'truth_7th_word_length',
    'truth_8th_word_length', 'truth_9th_word_length',
    'truth_10th_word_length', 'truth_11th_word_length',
    'truth_12th_word_length', 'truth_13th_word_length',
    'truth_14th_word_length', 'truth_15th_word_length',
    'truth_1th_word_probability', 'truth_2th_word_probability',
    'truth_3th_word_probability', 'truth_4th_word_probability',
    'truth_5th_word_probability', 'truth_6th_word_probability',
    'truth_7th_word_probability', 'truth_8th_word_probability',
    'truth_9th_word_probability', 'truth_10th_word_probability',
    'truth_11th_word_probability', 'truth_12th_word_probability',
    'truth_13th_word_probability', 'truth_14th_word_probability',
    'truth_15th_word_probability', 'truth_1th_word_probability_rank',
    'truth_2th_word_probability_rank', 'truth_3th_word_probability_rank',
    'truth_4th_word_probability_rank', 'truth_5th_word_probability_rank',
    'truth_6th_word_probability_rank', 'truth_7th_word_probability_rank',
    'truth_8th_word_probability_rank', 'truth_9th_word_probability_rank',
    'truth_10th_word_probability_rank', 'truth_11th_word_probability_rank',
    'truth_12th_word_probability_rank', 'truth_13th_word_probability_rank',
    'truth_14th_word_probability_rank', 'truth_15th_word_probability_rank',
    'truth_1th_word_best_match_score', 'truth_2th_word_best_match_score',
    'truth_3th_word_best_match_score', 'truth_4th_word_best_match_score',
    'truth_5th_word_best_match_score', 'truth_6th_word_best_match_score',
    'truth_7th_word_best_match_score', 'truth_8th_word_best_match_score',
    'truth_9th_word_best_match_score', 'truth_10th_word_best_match_score',
    'truth_11th_word_best_match_score', 'truth_12th_word_best_match_score',
    'truth_13th_word_best_match_score', 'truth_14th_word_best_match_score',
    'truth_15th_word_best_match_score', 'reconstructed_score'
]


# In[17]:


def generate_prediction(index, title_to_match):
    title_to_match_words = title_to_match.split(' ')
    
    all_nearest_found = get_nearest_matches(index)
    
    for matches_nearest in [all_nearest_found[0:10], all_nearest_found[10:100], all_nearest_found[100:]]:
        
        matches = [ground_truth_mapping[x] for x in matches_nearest]
        len_matches = len(matches)

        truth_company_name = [x['truth_company_name'] for x in matches]
        truth_number_of_words = [x['truth_number_of_words'] for x in matches]
        truth_company_name_words = [x['truth_company_name_words'] for x in matches]
        truth_number_of_characters = [x['truth_number_of_characters'] for x in matches]

        company_name = [title_to_match] * len_matches
        company_name_words = [title_to_match_words] * len_matches
        number_of_characters = [len(title_to_match)] * len_matches
        number_of_words = [len(title_to_match_words)] * len_matches
        levenshtein = list(map(lambda x: fuzz.ratio(x[0], x[1]), zip(truth_company_name, company_name)))

        extra_features = list(map(lambda x: construct_features(x[0], x[1], x[2]), zip(
            truth_number_of_words, 
            truth_company_name_words, 
            company_name)))

        features = np.array([
            number_of_characters,
            truth_number_of_characters,
            number_of_words,
            truth_number_of_words,
            levenshtein
        ])

        features = np.concatenate([features.T, extra_features], axis=1)

        d_test = xgb.DMatrix(features, feature_names=feature_names)
        predictions = model.predict(d_test)
        best_match_index = np.argmax(predictions)
        best_match = matches[best_match_index]['truth_company_name']
        best_match_id = matches_nearest[best_match_index]
        best_match_prediction = predictions[best_match_index]

        if best_match_prediction > 0.5:
            matched_so_far.append(index)
            save_prediction(index, title_to_match, best_match, best_match_id, best_match_prediction)
            break
    
    return


# In[18]:


get_ipython().run_cell_magic('time', '', "\n# 3rd Step\n\nexecutor = ProcessPoolExecutor(max_workers=3)\nthreads = [\n    executor.submit(generate_prediction, index, row['transformed_name']) \n    for index, row in test_data.loc[~test_data.index.isin(matched_so_far), :].iterrows()\n]\n\nrunning = sum([x.running() for x in threads])\nwhile running != 0:\n    time.sleep(0.5)\n    running = sum([x.running() for x in threads])")


# In[19]:


len(test_data.loc[~test_data.index.isin(matched_so_far), :])


# In[20]:


submission = pd.read_sql("SELECT test_id AS test_index, best_match_id AS company_id FROM predictions", connection)


# In[21]:


not_macthed_rows = test_data.loc[~test_data.index.isin(matched_so_far), :].copy(deep=True)
not_macthed_rows.index.name = 'test_index'
not_macthed_rows.reset_index(inplace=True)
not_macthed_rows.loc[:, 'company_id'] = -1
not_macthed_rows = not_macthed_rows.loc[:, ['test_index', 'company_id']].copy(deep=True)


# In[22]:


final_submission = pd.concat([not_macthed_rows, submission])
final_submission.sort_values(['test_index'], inplace=True)
final_submission.reset_index(drop=True, inplace=True)


# In[23]:


final_submission.to_csv('final_submission.csv', index=False, sep='|')


# In[24]:


final_submission['company_id'].nunique(), len(final_submission[final_submission['company_id'] != -1])


# In[ ]:




