import json
import sqlite3
import logging
import _pickle as pickle

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import xgboost as xgb

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import get_test_data, run_in_multi_processing_mode, transform_title
from doppelspeller.feature_engineering import FeatureEngineering


LOGGER = logging.getLogger(__name__)

# TODO: multiprocessing module can not pickle self.<attributes>, therefore, defining global variables!
# Try to avoid declaring these variables, maybe use pathos?
(CONNECTION, CURSOR, LSH_FOREST, GROUND_TRUTH,
 GROUND_TRUTH_MAPPING_REVERSED, GROUND_TRUTH_MAPPING, MODEL,
 WORDS_COUNTER, NUMBER_OF_WORDS) = (None, None, None, None, None, None, None, None, None)


class Prediction:
    def __init__(self):
        self.already_populated_required_data = False
        self.matched_so_far = []

    def _populate_required_data(self):
        if self.already_populated_required_data:
            return True

        global LSH_FOREST, MODEL, GROUND_TRUTH, GROUND_TRUTH_MAPPING, \
            GROUND_TRUTH_MAPPING_REVERSED, CONNECTION, CURSOR, \
            WORDS_COUNTER, NUMBER_OF_WORDS

        LOGGER.info('Reading LSH forest dump!')
        with open(s.LSH_FOREST_OUTPUT_FILE, 'rb') as fl:
            LSH_FOREST = pickle.load(fl)

        GROUND_TRUTH, WORDS_COUNTER, _ = FeatureEngineering.populate_required_data()

        GROUND_TRUTH.set_index(c.COLUMN_TITLE_ID, inplace=True)
        GROUND_TRUTH.sort_index(inplace=True)
        GROUND_TRUTH.loc[:, c.COLUMN_TITLE_ID] = GROUND_TRUTH.index

        self.connection = sqlite3.connect(s.SQLITE_DB)
        self.cursor = self.connection.cursor()
        CONNECTION = self.connection
        CURSOR = self.cursor

        GROUND_TRUTH_MAPPING = GROUND_TRUTH.set_index(c.COLUMN_TITLE_ID).copy(deep=True)
        GROUND_TRUTH_MAPPING = GROUND_TRUTH_MAPPING.to_dict()[c.COLUMN_TRANSFORMED_TITLE]
        GROUND_TRUTH_MAPPING = {
            k: {
                c.COLUMN_TRUTH_TITLE: v,
                c.COLUMN_TRUTH_WORDS: v.split(' '),
                c.COLUMN_TRUTH_NUMBER_OF_CHARACTERS: len(v),
                c.COLUMN_TRUTH_NUMBER_OF_WORDS: len(v.split(' ')),
            } for k, v in GROUND_TRUTH_MAPPING.items()
        }
        GROUND_TRUTH_MAPPING_REVERSED = {v[c.COLUMN_TRUTH_TITLE]: k for k, v in GROUND_TRUTH_MAPPING.items()}

        with open(s.MODEL_DUMP_FILE, 'rb') as fl:
            MODEL = pickle.load(fl)

        self.test_data = get_test_data()

        self.already_populated_required_data = True

    @staticmethod
    def _get_nearest_matches(test_id):
        query = f'SELECT matches from {s.SQLITE_NEIGHBOURS_TABLE} WHERE test_id={test_id}'
        CURSOR.execute(query)
        return json.loads(CURSOR.fetchone()[0])

    @staticmethod
    def _save_prediction(test_id, title_to_match, best_match, best_match_id, best_match_probability):
        query = f"INSERT INTO {s.SQLITE_PREDICTIONS_TABLE} VALUES ({test_id}, '{title_to_match}', " \
                f"'{best_match}', {best_match_id}, {best_match_probability});"
        CURSOR.execute(query)
        CONNECTION.commit()
        return True

    @staticmethod
    def _generate_single_prediction(index, title_to_match, find_closest=False):
        if find_closest:
            all_nearest_found = list(GROUND_TRUTH.index)
            matches_chunk = [all_nearest_found]
        else:
            all_nearest_found = Prediction._get_nearest_matches(index)
            matches_chunk = [all_nearest_found[0:10], all_nearest_found[10:100], all_nearest_found[100:]]

        for matches_nearest in matches_chunk:
            if not matches_nearest:
                continue

            prediction_features = np.zeros((len(matches_nearest),), dtype=s.FEATURES_TYPES)
            matches = []
            for matrix_index, match_index in enumerate(matches_nearest):
                match = GROUND_TRUTH_MAPPING[match_index][c.COLUMN_TRUTH_TITLE]
                matches.append(match)

                kind, title, truth_title, target = (
                    s.DISABLE_TRAIN_KIND_VALUE, title_to_match, match, s.DISABLE_TARGET_VALUE
                )
                prediction_features[matrix_index] = FeatureEngineering.construct_features(
                    kind, title, truth_title, target)

            prediction_features_set = np.array(prediction_features.tolist(), dtype=np.float16)
            features_names = list(prediction_features.dtype.names)
            del prediction_features

            d_test = xgb.DMatrix(prediction_features_set, feature_names=features_names)
            predictions = MODEL.predict(d_test)

            best_match_index = np.argmax(predictions)
            best_match = matches[best_match_index]
            best_match_id = matches_nearest[best_match_index]
            best_match_prediction = predictions[best_match_index]

            if find_closest:
                return best_match_id, best_match, best_match_prediction
            else:
                if best_match_prediction > s.PREDICTION_PROBABILITY_THRESHOLD:
                    Prediction._save_prediction(index, title_to_match, best_match, best_match_id, best_match_prediction)
                    break

        return True

    def _find_exact_matches(self):
        LOGGER.info('Finding exact matches!')

        self.test_data.loc[:, c.COLUMN_EXACT] = self.test_data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
            lambda x: GROUND_TRUTH_MAPPING_REVERSED.get(x, -2))

        test_data_filtered = self.test_data.loc[self.test_data[c.COLUMN_EXACT] != -2, :]
        for test_index, title_to_match, best_match_id in zip(
            test_data_filtered.index,
            test_data_filtered[c.COLUMN_TRANSFORMED_TITLE],
            test_data_filtered[c.COLUMN_EXACT]
        ):
            best_match = GROUND_TRUTH_MAPPING[best_match_id][c.COLUMN_TRUTH_TITLE]
            self._save_prediction(test_index, title_to_match, best_match, best_match_id, 1.0)

    def _find_close_matches(self):
        LOGGER.info('Finding very close matches!')

        remaining = self.test_data.loc[~self.test_data.index.isin(self.matched_so_far), :]
        remaining_count = len(remaining)
        count = 0
        for index, title_to_match in zip(remaining.index, remaining[c.COLUMN_TRANSFORMED_TITLE]):
            count += 1
            if not (count % 10000):
                LOGGER.info(f'Processed {count} of {remaining_count}...')

            best_match_ids = self._get_nearest_matches(index)

            if not best_match_ids:
                self._save_prediction(index, title_to_match, None, s.TRAIN_NOT_FOUND_VALUE, 0.0)
                continue

            matches = [GROUND_TRUTH_MAPPING[best_match_id][c.COLUMN_TRUTH_TITLE] for best_match_id in best_match_ids]
            ratios = [fuzz.ratio(best_match, title_to_match) for best_match in matches]
            arg_max = np.argmax(ratios)
            max_ratio = ratios[arg_max]
            best_match_id = best_match_ids[arg_max]
            if max_ratio > 94:
                best_match = matches[arg_max]
                self._save_prediction(index, title_to_match, best_match, best_match_id, 1.0)
            else:
                ratios = [fuzz.token_sort_ratio(best_match, title_to_match) for best_match in matches]
                arg_max = np.argmax(ratios)
                max_ratio = ratios[arg_max]
                best_match_id = best_match_ids[arg_max]
                if max_ratio > 94:
                    best_match = matches[arg_max]
                    self._save_prediction(index, title_to_match, best_match, best_match_id, 1.0)

    def _find_matches_using_model(self):
        LOGGER.info('Finding matches using the model!')

        remaining = self.test_data.loc[~self.test_data.index.isin(self.matched_so_far), :]
        all_args_kwargs = [([index, title_to_match], {})
                           for index, title_to_match in zip(remaining.index, remaining[c.COLUMN_TRANSFORMED_TITLE])]
        _ = run_in_multi_processing_mode(self._generate_single_prediction, all_args_kwargs)

    def _update_matched_so_far(self):
        so_far = pd.read_sql(
            f"SELECT test_id AS {c.COLUMN_TEST_INDEX} "
            f"FROM {s.SQLITE_PREDICTIONS_TABLE} WHERE best_match_id <> '{s.TRAIN_NOT_FOUND_VALUE}'", self.connection)
        self.matched_so_far = list(so_far[c.COLUMN_TEST_INDEX].unique())
        LOGGER.info(f'Matched {len(self.matched_so_far)} titles so far!')

    def _finalize_output(self):
        output = pd.read_sql(
            f"SELECT test_id AS {c.COLUMN_TEST_INDEX}, best_match_id AS {c.COLUMN_TITLE_ID} "
            f"FROM {s.SQLITE_PREDICTIONS_TABLE}", self.connection)

        not_matched_rows = self.test_data.loc[~self.test_data.index.isin(self.matched_so_far), :].copy(deep=True)
        not_matched_rows.index.name = c.COLUMN_TEST_INDEX
        not_matched_rows.reset_index(inplace=True)
        not_matched_rows.loc[:, c.COLUMN_TITLE_ID] = s.TRAIN_NOT_FOUND_VALUE
        not_matched_rows = not_matched_rows.loc[:, [c.COLUMN_TEST_INDEX, c.COLUMN_TITLE_ID]].copy(deep=True)

        final_output = pd.concat([not_matched_rows, output])
        final_output.sort_values([c.COLUMN_TEST_INDEX], inplace=True)
        final_output.reset_index(drop=True, inplace=True)

        final_output.to_csv(s.FINAL_OUTPUT_FILE, index=False, sep='|')

    def _prepare_output_database_table(self):
        # Creating the SQLite table
        self.cursor.execute(f"DROP TABLE IF EXISTS {s.SQLITE_PREDICTIONS_TABLE};")
        self.cursor.execute(f"CREATE TABLE {s.SQLITE_PREDICTIONS_TABLE} "
                            f"(test_id INTEGER, "
                            f"title_to_match TEXT, "
                            f"best_match TEXT, "
                            f"best_match_id INTEGER, "
                            f"best_match_probability FLOAT);")
        self.connection.commit()

    def extensive_search_single_title(self, title):
        self._populate_required_data()
        return self._generate_single_prediction(0, transform_title(title), find_closest=True)

    def process(self):
        self._populate_required_data()
        self._prepare_output_database_table()

        steps = [self._find_exact_matches, self._find_close_matches, self._find_matches_using_model]
        for step in steps:
            step()
            self._update_matched_so_far()

        self._finalize_output()

        return s.FINAL_OUTPUT_FILE
