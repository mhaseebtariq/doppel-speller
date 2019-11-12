#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import random
import math
import _pickle as pickle
import random
from collections import Counter
from itertools import product

import unicodedata
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from datasketch import MinHashLSHForest, MinHash


# In[2]:


SEQUENCES_OF = 3
NUM_PERM = 128


# In[3]:


get_ipython().run_cell_magic('time', '', '\n%run common.ipynb')


# In[4]:


ground_truth.head()


# In[5]:


get_ipython().run_cell_magic('time', '', "\nground_truth.loc[:, 'sequences'] = ground_truth.loc[:, 'transformed_name'].apply(\n    lambda x: get_sequences(x, SEQUENCES_OF)\n)")


# In[6]:


ground_truth.head()


# In[7]:


get_ipython().run_cell_magic('time', '', "\nforest = MinHashLSHForest(num_perm=NUM_PERM, l=32)\ncount = 0\nfor company_id, sequences in zip(ground_truth.loc[:, 'company_id'], ground_truth.loc[:, 'sequences']):\n    if not (count % 50000):\n        print(count)\n    count += 1\n    minhash = MinHash(num_perm=NUM_PERM)\n    _ = [minhash.update(str(x).encode('utf8')) for x in sequences]\n    forest.add(company_id, minhash)")


# In[8]:


get_ipython().run_cell_magic('time', '', '\nforest.index()')


# In[9]:


# with open('lsh_forest.dump', 'wb') as fl:
#     pickle.dump(forest, fl)


# In[10]:


# with open('lsh_forest.dump', 'rb') as fl:
#     forest = pickle.load(fl)


# In[11]:


train_data = pd.read_csv('STrain.csv', delimiter='|')
train_data.loc[:, 'transformed_name'] = train_data.loc[:, 'name'].apply(lambda x: convert_text(x))
train_data.loc[:, 'sequences'] = train_data.loc[:, 'transformed_name'].apply(
    lambda x: get_sequences(x, SEQUENCES_OF)
)


# In[12]:


get_ipython().run_cell_magic('time', '', "\ntrain_data.loc[:, 'sequences_hash'] = train_data.loc[:, 'sequences'].apply(\n    lambda x: get_minhash(x, NUM_PERM))")


# In[13]:


get_ipython().run_cell_magic('time', '', "\n### FOR GENERATING TRAINING DATA ###\n\nin_top = 10\nsimilar = {-1: {}}\nfound, not_found = [], []\n\nfor index, row in train_data.iterrows():\n    company_id = row['company_id']\n    test = row['transformed_name']\n    sequences_hash = row['sequences_hash']\n    train_index = row['train_index']\n    matches = forest.query(sequences_hash, in_top)\n\n    if company_id == -1:\n        similar[company_id][train_index] = matches\n        continue\n    else:\n        similar[company_id] = matches\n\n    if company_id in matches:\n        found.append(company_id)\n    else:\n        if len(similar[company_id]) == in_top: \n            similar[company_id].pop()\n        similar[company_id].append(company_id)\n        not_found.append(company_id)\n\n    if not(index % 10000):\n        print(index, round((len(similar)/(index+1))*100, 2))\n\nprint(index, round((len(similar)/(index+1))*100, 2))")


# In[14]:


with open('similar_programs_{}.dump'.format(in_top), 'wb') as fl:
    pickle.dump(similar, fl)


# ## Generate more training data

# In[15]:


ground_truth.loc[:, 'number_of_chars'] = ground_truth.loc[:, 'transformed_name'].apply(
    lambda x: len(x)
)


# In[16]:


generated_training_data = ground_truth[ground_truth['number_of_chars'] > 9].copy(deep=True)


# In[17]:


keyboard_cartesian = {'q': {'x':0, 'y':0}, 'w': {'x':1, 'y':0}, 'e': {'x':2, 'y':0}, 'r': {'x':3, 'y':0}, 't': {'x':4, 'y':0}, 'y': {'x':5, 'y':0}, 'u': {'x':6, 'y':0}, 'i': {'x':7, 'y':0}, 'o': {'x':8, 'y':0}, 'p': {'x':9, 'y':0}, 'a': {'x':0, 'y':1},'z': {'x':0, 'y':2},'s': {'x':1, 'y':1},'x': {'x':1, 'y':2},'d': {'x':2, 'y':1},'c': {'x':2, 'y':2}, 'f': {'x':3, 'y':1}, 'b': {'x':4, 'y':2}, 'm': {'x':5, 'y':2}, 'j': {'x':6, 'y':1}, 'g': {'x':4, 'y':1}, 'h': {'x':5, 'y':1}, 'j': {'x':6, 'y':1}, 'k': {'x':7, 'y':1}, 'l': {'x':8, 'y':1}, 'v': {'x':3, 'y':2}, 'n': {'x':5, 'y':2}, }


# In[18]:


def euclidean_distance(a,b):
    X = (keyboard_cartesian[a]['x'] - keyboard_cartesian[b]['x'])**2
    Y = (keyboard_cartesian[a]['y'] - keyboard_cartesian[b]['y'])**2
    return math.sqrt(X+Y)


# In[19]:


neighbours = {}
for i in keyboard_cartesian.keys():
    for j in keyboard_cartesian.keys():
        if i == j:
            continue
        distance = euclidean_distance(i, j)
        if distance <= 1:
            if i in neighbours:
                neighbours[i].add(j)
            else:
                neighbours[i] = {j}
            
            if j in neighbours:
                neighbours[j].add(i)
            else:
                neighbours[j] = {i}


# In[20]:


neighbours = {k: list(v) for k, v in neighbours.items()}


# In[21]:


def remove_letter(x, length):
    index = random.randint(0, length-1)
    letter = x[index]
    count = 0
    while letter == ' ':
        count += 1
        if count > 10:
            return x
        index = random.randint(0, length-1)
        letter = x[index]
    return x[:index] + x[index+1:]

def add_letter(x, length):
    index = random.randint(0, length-1)
    letter = x[index]
    count = 0
    while letter in ' 0123456789':
        count += 1
        if count > 10:
            return x
        index = random.randint(0, length-1)
        letter = x[index]
    to_add_options = neighbours[letter]
    return x[:index] + random.choice(to_add_options) + x[index:]

def replace_letter(x, length):
    index = random.randint(0, length-1)
    letter = x[index]
    count = 0
    while letter in ' 0123456789':
        count += 1
        if count > 10:
            return x
        index = random.randint(0, length-1)
        letter = x[index]
    
    return x[:index] + random.choice(neighbours[x[index]]) + x[index+1:]

def add_space(x, length):
    index = random.randint(1, length-1)
    check = any([x[index] == ' ', x[index-1: index] in ('', ' '), x[index+1: index+2] in ('', ' ')])
    count = 0
    while check:
        count += 1
        if count > 10:
            return x
        index = random.randint(1, length-1)
        check = any([x[index] == ' ', x[index-1: index] in ('', ' '), x[index+1: index+2] in ('', ' ')])
    
    return x[:index] + ' ' + x[index:]

def remove_space(x, length):
    list_x = list(x)
    space_indexes = [index for index, item in enumerate(list_x) if item == ' ']
    if not space_indexes:
        return x
    to_remove = random.choice(space_indexes)
    del list_x[to_remove]

    return ''.join(list_x)


# In[22]:


def generate_misspelled_name(word):
    new_word = str(word)
    functions = [random.choice([add_letter, remove_letter]), replace_letter, random.choice([add_space, remove_space])]
    functions_selected = random.sample(functions, random.randint(1, 2))
    for func in functions_selected:
        new_word = func(new_word, len(new_word))
    return new_word


# In[23]:


get_ipython().run_cell_magic('time', '', "\ngenerated_training_data.loc[:, 'generated_misspelled'] = generated_training_data.loc[:, 'transformed_name'].apply(\n    lambda x: generate_misspelled_name(x)\n)")


# In[24]:


generated_training_data.loc[:, ['generated_misspelled', 'transformed_name']].reset_index().to_pickle(
    'generated_training_data.dump')


# In[ ]:




