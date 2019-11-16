import time
import json
import sqlite3
import logging
import _pickle as pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import get_ground_truth, get_min_hash, get_test_data


LOGGER = logging.getLogger(__name__)
CONNECTION, CURSOR, LSH_FOREST, GROUND_TRUTH = None, None, None, None


def populate_required_data():
    global LSH_FOREST, GROUND_TRUTH, CONNECTION, CURSOR

    LOGGER.info('Reading LSH forest dump!')
    with open(s.LSH_FOREST_OUTPUT_FILE, 'rb') as fl:
        LSH_FOREST = pickle.load(fl)

    GROUND_TRUTH = get_ground_truth()
    GROUND_TRUTH.set_index(c.COLUMN_TITLE_ID, inplace=True)
    GROUND_TRUTH.sort_index(inplace=True)
    GROUND_TRUTH.loc[:, c.COLUMN_TITLE_ID] = GROUND_TRUTH.index

    CONNECTION = sqlite3.connect(s.SQLITE_DB)
    CURSOR = CONNECTION.cursor()

    return True


def save_nearest_matches(test_index, test_sequences):
    min_hash = get_min_hash(test_sequences, num_perm=s.NUMBER_OF_PERMUTATIONS)
    nearest_neighbours = LSH_FOREST.query(min_hash, s.FETCH_NEAREST_N_IN_FOREST)

    if (not nearest_neighbours) or (not test_sequences):
        CURSOR.execute(f"INSERT INTO {s.SQLITE_NEIGHBOURS_TABLE} "
                       f"(test_id, matches) values ({test_index}, '{json.dumps([])}')")
        CONNECTION.commit()
        return

    nearest_neighbours = GROUND_TRUTH.loc[GROUND_TRUTH.index.isin(nearest_neighbours), :]
    distance_indexes = np.argsort(list(
        map(lambda x: -len(x.intersection(test_sequences)) / len(x.union(test_sequences)),
            nearest_neighbours[c.COLUMN_SEQUENCES].values))
    )[:s.TOP_N_RESULTS_IN_FOREST]

    matches = [int(x) for x in nearest_neighbours[c.COLUMN_TITLE_ID].values[distance_indexes]]

    CURSOR.execute(f"INSERT INTO {s.SQLITE_NEIGHBOURS_TABLE} (test_id, matches) values "
                   f"({test_index}, '{json.dumps(matches)}')")
    CONNECTION.commit()

    return True


def prepare_predictions_data():
    _ = populate_required_data()
    test_data = get_test_data()

    # Drop table
    CURSOR.execute(f"DROP TABLE IF EXISTS {s.SQLITE_NEIGHBOURS_TABLE};")

    # Create table
    CURSOR.execute(f"CREATE TABLE {s.SQLITE_NEIGHBOURS_TABLE} (test_id INTEGER, matches TEXT);")

    # Save (commit) the changes
    CONNECTION.commit()

    executor = ProcessPoolExecutor(max_workers=3)
    threads = [
        executor.submit(save_nearest_matches, index, row[c.COLUMN_SEQUENCES]) for index, row in test_data.iterrows()
    ]

    running = sum([x.running() for x in threads])
    while running != 0:
        time.sleep(0.5)
        running = sum([x.running() for x in threads])

    CURSOR.execute(f"CREATE UNIQUE INDEX id_index ON {s.SQLITE_NEIGHBOURS_TABLE} (test_id);")
    CONNECTION.commit()

    CURSOR.execute(f"SELECT COUNT(*) FROM {s.SQLITE_NEIGHBOURS_TABLE} LIMIT 1")
    LOGGER.info(f"Inserted {CURSOR.fetchone()[0]} rows in to table {s.SQLITE_NEIGHBOURS_TABLE}")
