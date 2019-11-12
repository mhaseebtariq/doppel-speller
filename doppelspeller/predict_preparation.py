#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import json
import sqlite3
import _pickle as pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from datasketch import MinHashLSHForest, MinHash


# In[2]:


SEQUENCES_OF = 3
NUM_PERM = 128


# In[3]:


get_ipython().run_line_magic('run', 'common.ipynb')


# In[4]:


ground_truth.set_index('company_id', inplace=True)
ground_truth.sort_index(inplace=True)
ground_truth.loc[:, 'company_id'] = ground_truth.index


# In[5]:


get_ipython().run_cell_magic('time', '', "\nground_truth.loc[:, 'sequences'] = ground_truth.loc[:, 'transformed_name'].apply(\n    lambda x: get_sequences(x, SEQUENCES_OF)\n)")


# In[6]:


get_ipython().run_cell_magic('time', '', "\ntest_data = pd.read_csv('STest.csv', delimiter='|')\ntest_data.loc[:, 'transformed_name'] = test_data.loc[:, 'name'].apply(lambda x: convert_text(x))\ndel test_data['name']\ndel test_data['test_index']\ntest_data.loc[:, 'sequences'] = test_data.loc[:, 'transformed_name'].apply(\n    lambda x: get_sequences(x, SEQUENCES_OF)\n)")


# In[7]:


get_ipython().run_cell_magic('time', '', "\nwith open('lsh_forest.dump', 'rb') as fl:\n    lsh_forest = pickle.load(fl)")


# In[8]:


connection = sqlite3.connect('data.db')
cursor = connection.cursor()

# Drop table
cursor.execute('''DROP TABLE IF EXISTS neighbours;''')

# Create table
cursor.execute('''CREATE TABLE neighbours (test_id INTEGER, matches TEXT);''')

# Save (commit) the changes
connection.commit()


# In[9]:


def get_nearest_matches(test_index, test_sequences, nearest_in_forest=10000, n_matches=1000):
    minhash = get_minhash(test_sequences, num_perm=NUM_PERM)
    nearest_neighbours = lsh_forest.query(minhash, nearest_in_forest)
    
    if (not nearest_neighbours) or (not test_sequences):
        cursor.execute("INSERT INTO neighbours (test_id, matches) values ({}, '{}')".format(
            test_index, json.dumps([])))
        connection.commit()
        return
    
    nearest_neighbours = ground_truth.loc[ground_truth.index.isin(nearest_neighbours), :]
    jaccards_indexes = np.argsort(list(
        map(lambda x: -len(x.intersection(test_sequences))/len(x.union(test_sequences)),
            nearest_neighbours['sequences'].values))
    )[:n_matches]
        
    matches = [int(x) for x in nearest_neighbours['company_id'].values[jaccards_indexes]]

    cursor.execute("INSERT INTO neighbours (test_id, matches) values ({}, '{}')".format(
        test_index, json.dumps(matches)))
    connection.commit()
    
    return


# In[10]:


get_ipython().run_cell_magic('time', '', "\nexecutor = ProcessPoolExecutor(max_workers=3)\nthreads = [\n    executor.submit(get_nearest_matches, index, row['sequences']) for index, row in test_data.iterrows()\n]")


# In[11]:


get_ipython().run_cell_magic('time', '', '\nrunning = sum([x.running() for x in threads])\nwhile running != 0:\n    time.sleep(0.5)\n    running = sum([x.running() for x in threads])')


# In[12]:


get_ipython().run_cell_magic('time', '', '\ncursor.execute("CREATE UNIQUE INDEX id_index ON neighbours (test_id);")\nconnection.commit()')


# In[13]:


cursor.execute("SELECT COUNT(*) FROM neighbours LIMIT 1")
print(cursor.fetchone()[0])


# In[ ]:




