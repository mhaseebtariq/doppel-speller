import logging
import _pickle as pickle

import pandas as pd
import numpy as np
import xgboost as xgb

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import load_processed_test_data, get_words_counter, transform_title
from doppelspeller.feature_engineering import FeatureEngineering, levenshtein_ratio, levenshtein_token_sort_ratio


LOGGER = logging.getLogger(__name__)


class Prediction:
    def __init__(self):
        self.processed_data = load_processed_test_data()

        self.data = self.processed_data[c.DATA_TYPE_TEST]
        self.truth_data = self._set_index_truth_data(self.processed_data[c.DATA_TYPE_TRUTH])
        self.nearest_matches = self.processed_data[c.DATA_TYPE_NEAREST_TEST]

        self.words_counter = get_words_counter(self.truth_data)
        self.number_of_titles = len(self.truth_data)

        self.truth_data_mapping, self.truth_data_mapping_reversed = self. _get_truth_data_mappings(self.truth_data)

        self.model = self._load_model()
        self.predictions = self._initiate_predictions_data(list(self.data[c.COLUMN_TEST_INDEX]))
        self.feature_engineering = FeatureEngineering(c.DATA_TYPE_TEST)

    @staticmethod
    def _initiate_predictions_data(test_index):
        predictions = pd.DataFrame(test_index, columns=[c.COLUMN_TEST_INDEX])
        predictions.loc[:, c.COLUMN_TITLE_TO_MATCH] = ""
        predictions.loc[:, c.COLUMN_BEST_MATCH] = ""
        predictions.loc[:, c.COLUMN_BEST_MATCH_ID] = s.TRAIN_NOT_FOUND_VALUE
        predictions.loc[:, c.COLUMN_BEST_MATCH_PROBABILITY] = 0.0
        return predictions

    @staticmethod
    def _set_index_truth_data(truth_data):
        truth_data.set_index(c.COLUMN_TITLE_ID, inplace=True)
        truth_data.sort_index(inplace=True)
        truth_data.loc[:, c.COLUMN_TITLE_ID] = truth_data.index
        return truth_data

    @staticmethod
    def _get_truth_data_mappings(truth_data):
        truth_data_mapping = truth_data.set_index(c.COLUMN_TITLE_ID).copy(deep=True)
        truth_data_mapping = truth_data_mapping.to_dict()[c.COLUMN_TRANSFORMED_TITLE]
        truth_data_mapping = {
            k: {
                c.COLUMN_TRUTH_TITLE: v,
                c.COLUMN_TRUTH_WORDS: v.split(' '),
                c.COLUMN_TRUTH_NUMBER_OF_CHARACTERS: len(v),
                c.COLUMN_TRUTH_NUMBER_OF_WORDS: len(v.split(' ')),
            } for k, v in truth_data_mapping.items()
        }
        truth_data_mapping_reversed = {v[c.COLUMN_TRUTH_TITLE]: k for k, v in truth_data_mapping.items()}

        return truth_data_mapping, truth_data_mapping_reversed

    @staticmethod
    def _load_model():
        with open(s.MODEL_DUMP_FILE, 'rb') as fl:
            return pickle.load(fl)

    def _get_nearest_matches(self, test_id):
        return self.nearest_matches[test_id]

    def _save_prediction(self, test_index, title_to_match, best_match, best_match_id, best_match_probability):
        filter_ = self.predictions[c.COLUMN_TEST_INDEX] == test_index
        to_update_columns = [
            c.COLUMN_TITLE_TO_MATCH,
            c.COLUMN_BEST_MATCH,
            c.COLUMN_BEST_MATCH_ID,
            c.COLUMN_BEST_MATCH_PROBABILITY,
        ]
        self.predictions.loc[filter_, to_update_columns] = [
            title_to_match,
            best_match,
            best_match_id,
            best_match_probability,
        ]

    def _generate_single_prediction(self, index, title_to_match, find_closest=False):
        if find_closest:
            all_nearest_found = list(self.truth_data.index)
            matches_chunk = [all_nearest_found]
        else:
            all_nearest_found = self._get_nearest_matches(index)
            matches_chunk = [all_nearest_found[0:10], all_nearest_found[10:100]]

        for matches_nearest in matches_chunk:
            prediction_features = np.zeros((len(matches_nearest),), dtype=s.FEATURES_TYPES)
            matches = []
            for matrix_index, match_index in enumerate(matches_nearest):
                match = self.truth_data_mapping[match_index][c.COLUMN_TRUTH_TITLE]
                matches.append(match)

                kind, title, truth_title, target = (
                    s.DISABLE_TRAIN_KIND_VALUE, title_to_match, match, s.DISABLE_TARGET_VALUE
                )
                prediction_features[matrix_index] = self.feature_engineering.construct_features(
                    kind, title, truth_title, target)

            prediction_features_set = np.array(prediction_features.tolist(), dtype=np.float16)
            features_names = list(prediction_features.dtype.names)
            del prediction_features

            d_test = xgb.DMatrix(prediction_features_set, feature_names=features_names)
            predictions = self.model.predict(d_test)

            best_match_index = np.argmax(predictions)
            best_match = matches[best_match_index]
            best_match_id = matches_nearest[best_match_index]
            best_match_prediction = predictions[best_match_index]

            if find_closest:
                return best_match_id, best_match, best_match_prediction
            else:
                if best_match_prediction > s.PREDICTION_PROBABILITY_THRESHOLD:
                    self._save_prediction(index, title_to_match, best_match, best_match_id, best_match_prediction)
                    break

        return True

    def _find_exact_matches(self):
        LOGGER.info('Finding exact matches!')

        exact_value_flag = -2
        self.data.loc[:, c.COLUMN_EXACT] = self.data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
            lambda x: self.truth_data_mapping_reversed.get(x, exact_value_flag))

        test_data_filtered = self.data.loc[self.data[c.COLUMN_EXACT] != exact_value_flag, :]
        for test_index, title_to_match, best_match_id in zip(
            test_data_filtered.index,
            test_data_filtered[c.COLUMN_TRANSFORMED_TITLE],
            test_data_filtered[c.COLUMN_EXACT]
        ):
            best_match = self.truth_data_mapping[best_match_id][c.COLUMN_TRUTH_TITLE]
            self._save_prediction(test_index, title_to_match, best_match, best_match_id, 1.0)

    def _find_close_matches(self):
        LOGGER.info('Finding very close matches!')

        remaining = self.data.loc[~self.data.index.isin(self.matched_so_far), :]
        remaining_count = len(remaining)
        count = 0
        for index, title_to_match in zip(remaining.index, remaining[c.COLUMN_TRANSFORMED_TITLE]):
            count += 1
            if not (count % 1000):
                LOGGER.info(f'Processed {count} of {remaining_count}...')

            best_match_ids = self._get_nearest_matches(index)[:10]

            matches = [self.truth_data_mapping[best_match_id][c.COLUMN_TRUTH_TITLE]
                       for best_match_id in best_match_ids]
            ratios = [levenshtein_ratio(best_match, title_to_match) for best_match in matches]
            arg_max = np.argmax(ratios)
            max_ratio = ratios[arg_max]
            best_match_id = best_match_ids[arg_max]
            if max_ratio > 94:
                best_match = matches[arg_max]
                self._save_prediction(index, title_to_match, best_match, best_match_id, 1.0)
            else:
                ratios = [levenshtein_token_sort_ratio(best_match, title_to_match) for best_match in matches]
                arg_max = np.argmax(ratios)
                max_ratio = ratios[arg_max]
                best_match_id = best_match_ids[arg_max]
                if max_ratio > 94:
                    best_match = matches[arg_max]
                    self._save_prediction(index, title_to_match, best_match, best_match_id, 1.0)

    def _find_matches_using_model(self):
        LOGGER.info('Finding matches using the model!')

        remaining = self.data.loc[~self.data.index.isin(self.matched_so_far), :]
        number_of_rows_remaining = len(remaining)
        for count, (index, title_to_match) in enumerate(zip(remaining.index, remaining[c.COLUMN_TRANSFORMED_TITLE])):
            if not((count+1) % 1000):
                print(f'Processed {count+1} of {number_of_rows_remaining}!')
            self._generate_single_prediction(index, title_to_match)

    def _update_matched_so_far(self):
        filter_ = self.predictions[c.COLUMN_BEST_MATCH_ID] != s.TRAIN_NOT_FOUND_VALUE
        self.matched_so_far = list(self.predictions.loc[filter_, c.COLUMN_TEST_INDEX])
        LOGGER.info(f'Matched {len(self.matched_so_far)} titles so far!')

    def _finalize_output(self):
        predictions = self.predictions.loc[:, [c.COLUMN_TEST_INDEX, c.COLUMN_BEST_MATCH_ID]].copy(deep=True)
        predictions.rename(columns={c.COLUMN_BEST_MATCH_ID: c.COLUMN_TITLE_ID}, inplace=True)
        predictions.to_csv(s.FINAL_OUTPUT_FILE, index=False, sep=s.TEST_FILE_DELIMITER)

    def extensive_search_single_title(self, title):
        return self._generate_single_prediction(0, transform_title(title), find_closest=True)

    def process(self):
        steps = [self._find_exact_matches, self._find_close_matches, self._find_matches_using_model]
        for step in steps:
            step()
            self._update_matched_so_far()

        self._finalize_output()

        return s.FINAL_OUTPUT_FILE
