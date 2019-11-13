import time
import json
import sqlite3
import _pickle as pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SEQUENCES_OF = 3
NUM_PERM = 128


def predict_preparation():

    ground_truth.set_index('company_id', inplace=True)
    ground_truth.sort_index(inplace=True)
    ground_truth.loc[:, 'company_id'] = ground_truth.index

    ground_truth.loc[:, 'sequences'] = ground_truth.loc[:, 'transformed_name'].apply(
        lambda x: get_sequences(x, SEQUENCES_OF)
    )

    test_data = pd.read_csv('STest.csv', delimiter='|')
    test_data.loc[:, 'transformed_name'] = test_data.loc[:, 'name'].apply(lambda x: convert_text(x))
    del test_data['name']
    del test_data['test_index']
    test_data.loc[:, 'sequences'] = test_data.loc[:, 'transformed_name'].apply(
        lambda x: get_sequences(x, SEQUENCES_OF)
    )

    with open('lsh_forest.dump', 'rb') as fl:
        lsh_forest = pickle.load(fl)

    connection = sqlite3.connect('data.db')
    cursor = connection.cursor()

    # Drop table
    cursor.execute('''DROP TABLE IF EXISTS neighbours;''')

    # Create table
    cursor.execute('''CREATE TABLE neighbours (test_id INTEGER, matches TEXT);''')

    # Save (commit) the changes
    connection.commit()


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
            map(lambda x: -len(x.intersection(test_sequences)) / len(x.union(test_sequences)),
                nearest_neighbours['sequences'].values))
        )[:n_matches]

        matches = [int(x) for x in nearest_neighbours['company_id'].values[jaccards_indexes]]

        cursor.execute("INSERT INTO neighbours (test_id, matches) values ({}, '{}')".format(
            test_index, json.dumps(matches)))
        connection.commit()

        return


    executor = ProcessPoolExecutor(max_workers=3)
    threads = [
        executor.submit(get_nearest_matches, index, row['sequences']) for index, row in test_data.iterrows()
    ]

    running = sum([x.running() for x in threads])
    while running != 0:
        time.sleep(0.5)
        running = sum([x.running() for x in threads])

    cursor.execute("CREATE UNIQUE INDEX id_index ON neighbours (test_id);")
    connection.commit()

    cursor.execute("SELECT COUNT(*) FROM neighbours LIMIT 1")
    print(cursor.fetchone()[0])
