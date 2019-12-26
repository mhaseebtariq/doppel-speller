import json
import sqlite3
import logging
import _pickle as pickle

import numpy as np

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import get_ground_truth, get_min_hash, get_test_data, run_in_multi_processing_mode


LOGGER = logging.getLogger(__name__)

LSH_FOREST, GROUND_TRUTH, CONNECTION, CURSOR = None, None, None, None


class PrePredictionData:
    def _populate_pre_requisite_data(self):
        global LSH_FOREST, GROUND_TRUTH, CONNECTION, CURSOR

        LOGGER.info('Reading LSH forest dump!')
        with open(s.LSH_FOREST_OUTPUT_FILE, 'rb') as fl:
            self.lsh_forest = pickle.load(fl)
            LSH_FOREST = self.lsh_forest

        self.ground_truth = get_ground_truth()
        self.ground_truth.set_index(c.COLUMN_TITLE_ID, inplace=True)
        self.ground_truth.sort_index(inplace=True)
        self.ground_truth.loc[:, c.COLUMN_TITLE_ID] = self.ground_truth.index
        self.ground_truth = self.ground_truth
        GROUND_TRUTH = self.ground_truth

        self.test_data = get_test_data()

        self.connection = sqlite3.connect(s.SQLITE_DB)
        self.cursor = self.connection.cursor()
        CONNECTION, CURSOR = self.connection, self.cursor

    @staticmethod
    def _save_nearest_matches(test_index, test_sequences):
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

        matches = [x for x in nearest_neighbours[c.COLUMN_TITLE_ID].values[distance_indexes]]

        CURSOR.execute(f"INSERT INTO {s.SQLITE_NEIGHBOURS_TABLE} (test_id, matches) values "
                       f"({test_index}, '{json.dumps(matches)}')")
        CONNECTION.commit()

        return True

    def _prepare_output_database_table(self):
        # Creating the SQLite table
        self.cursor.execute(f"DROP TABLE IF EXISTS {s.SQLITE_NEIGHBOURS_TABLE};")
        self.cursor.execute(f"DROP INDEX IF EXISTS neighbours_id_index;")
        self.cursor.execute(f"CREATE TABLE {s.SQLITE_NEIGHBOURS_TABLE} (test_id INTEGER, matches TEXT);")
        self.connection.commit()

    def process(self):
        self._populate_pre_requisite_data()
        self._prepare_output_database_table()

        all_args_kwargs = [([test_index, sequences], {})
                           for test_index, sequences in zip(self.test_data.index, self.test_data[c.COLUMN_SEQUENCES])]
        _ = run_in_multi_processing_mode(self._save_nearest_matches, all_args_kwargs)

        # Creating index on the SQLite table
        self.cursor.execute(f"CREATE UNIQUE INDEX neighbours_id_index ON {s.SQLITE_NEIGHBOURS_TABLE} (test_id);")
        self.connection.commit()

        self.cursor.execute(f"SELECT COUNT(*) FROM {s.SQLITE_NEIGHBOURS_TABLE} LIMIT 1")
        LOGGER.info(f"Inserted {self.cursor.fetchone()[0]} rows in to table {s.SQLITE_NEIGHBOURS_TABLE}")
