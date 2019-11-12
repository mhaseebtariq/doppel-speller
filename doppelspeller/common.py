#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import re
from collections import Counter
from itertools import product

import unicodedata
import pandas as pd
import numpy as np


# In[7]:


SUBSTITUTE_REGEX = re.compile(r'[\s\-]+')
KEEP_REGEX = re.compile(r'[a-zA-Z0-9\s]')


def convert_text(text):
    # Remove accents
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8').lower()
    # Extract only alphanumeric characters / convert to lower case
    text = SUBSTITUTE_REGEX.sub(' ', text).strip()
    return ''.join(KEEP_REGEX.findall(text))


# In[ ]:


ground_truth = pd.read_csv('G.csv', delimiter='|')
ground_truth.loc[:, 'transformed_name'] = ground_truth.loc[:, 'name'].apply(lambda x: convert_text(x))
ground_truth.loc[:, 'words'] = ground_truth.loc[:, 'transformed_name'].apply(lambda x: x.split(' '))
ground_truth.loc[:, 'number_of_words'] = ground_truth.loc[:, 'words'].apply(lambda x: len(x))

WORDS = [ x for y in ground_truth.loc[:, 'words'] for x in y]
NUMBER_OF_WORDS = len(WORDS)
WORDS_COUNTER = Counter(WORDS)


# In[2]:


def get_sequences(words, sequences_of): 
    return set([words[i:i+sequences_of] for i in range(len(words)) if len(words[i:i+sequences_of]) == sequences_of])


# In[3]:


def word_probablility(word):
    return WORDS_COUNTER[word] / NUMBER_OF_WORDS


# In[1]:


def get_minhash(title, num_perm):
    minhash = MinHash(num_perm=num_perm)
    _ = [minhash.update(str(x).encode('utf8')) for x in title]
    return minhash


# In[ ]:


def construct_features(num_truth_words, truth_words, title_to_match, n=15):
    title_to_match = title_to_match.replace(' ', '')
    range_title_to_match = range(len(title_to_match))
    
    extra_nans = [np.nan] * (n - num_truth_words)
    
    truth_words = truth_words[:n]
    
    word_lengths = [len(x) for x in truth_words]
    word_probabilities = [word_probablility(x) for x in truth_words]
    word_probabilities_ranks = list(np.argsort(word_probabilities).argsort() + 1)
    
    best_scores = []
    constructed_title = []
    for length_truth_word, truth_word in zip(word_lengths, truth_words):
        possible_words = list(set([title_to_match[i:i+length_truth_word] for i in range_title_to_match]))
        ratios = [fuzz.ratio(truth_word, i) for i in possible_words]
        arg_max = np.argmax(ratios)
        best_score = ratios[arg_max]
        best_score_match = possible_words[arg_max]
        constructed_title.append(best_score_match)
        best_scores.append(best_score)
    
    reconstructed_score = fuzz.ratio(' '.join(constructed_title), ' '.join(truth_words))
    return (
          (word_lengths + extra_nans)
        + (word_probabilities + extra_nans)
        + (word_probabilities_ranks + extra_nans)
        + (best_scores + extra_nans)
        + [reconstructed_score]
    )

