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
        self.matched_so_far = []

    @staticmethod
    def _initiate_predictions_data(test_index):
        predictions = pd.DataFrame(index=test_index)
        predictions.index.name = c.COLUMN_TEST_INDEX
        predictions.sort_index(inplace=True)
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
        truth_data_mapping = {k: v for k, v in truth_data_mapping.items()}
        truth_data_mapping_reversed = {v: k for k, v in truth_data_mapping.items()}

        return truth_data_mapping, truth_data_mapping_reversed

    @staticmethod
    def _load_model():
        with open(s.MODEL_DUMP_FILE, 'rb') as fl:
            return pickle.load(fl)

    def _get_nearest_matches(self, test_id):
        return self.nearest_matches[test_id]

    def _save_prediction(self, test_index, title_to_match, best_match, best_match_id, best_match_probability):
        to_update_columns = [
            c.COLUMN_TITLE_TO_MATCH,
            c.COLUMN_BEST_MATCH,
            c.COLUMN_BEST_MATCH_ID,
            c.COLUMN_BEST_MATCH_PROBABILITY,
        ]
        self.predictions.loc[test_index, to_update_columns] = [
            title_to_match,
            best_match,
            best_match_id,
            best_match_probability,
        ]

    def _predict(self, prediction_features):
        prediction_features_set = np.array(prediction_features.tolist(), dtype=np.float16)
        features_names = list(prediction_features.dtype.names)
        del prediction_features

        d_test = xgb.DMatrix(prediction_features_set, feature_names=features_names)
        return self.model.predict(d_test)

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
            best_match = self.truth_data_mapping[best_match_id]
            self._save_prediction(test_index, title_to_match, best_match, best_match_id, 1.0)

    def _get_nearest_match_title_id(self, index, test_index, number_of_matches):
        index = index % number_of_matches
        return self.nearest_matches[test_index][index]

    def _get_nearest_match_title(self, match_id):
        return self.truth_data_mapping[match_id]

    def _combine_titles_with_matches(self, number_of_matches):
        remaining = self.data.loc[~self.data.index.isin(self.matched_so_far), :].copy(deep=True)
        remaining = remaining.loc[:, [c.COLUMN_TEST_INDEX, c.COLUMN_TRANSFORMED_TITLE]]

        if len(self.nearest_matches[0]) < number_of_matches:
            raise Exception('len(self.nearest_matches[0]) < number_of_matches')

        remaining = remaining.loc[remaining.index.repeat(number_of_matches)]
        remaining.reset_index(drop=True, inplace=True)

        matches_title_ids = map(lambda x, y: self._get_nearest_match_title_id(x, y, number_of_matches),
                                remaining.index, remaining[c.COLUMN_TEST_INDEX])
        remaining.loc[:, c.COLUMN_MATCH_TITLE_ID] = list(matches_title_ids)
        remaining.loc[:, c.COLUMN_MATCH_TRANSFORMED_TITLE] = remaining[c.COLUMN_MATCH_TITLE_ID].apply(
            self._get_nearest_match_title)

        return remaining

    def _find_close_matches(self, token_sort):
        LOGGER.info(f'Finding very close matches - token_sort={token_sort}!')

        levenshtein_ratio_function = levenshtein_ratio
        if token_sort:
            levenshtein_ratio_function = levenshtein_token_sort_ratio

        remaining = self._combine_titles_with_matches(10)

        matches_ratios = map(lambda x, y: levenshtein_ratio_function(x, y),
                             remaining[c.COLUMN_TRANSFORMED_TITLE], remaining[c.COLUMN_MATCH_TRANSFORMED_TITLE])
        remaining.loc[:, c.COLUMN_LEVENSHTEIN_RATIO] = list(matches_ratios)

        indexes_with_max_ratios = remaining.groupby(
            [c.COLUMN_TEST_INDEX])[c.COLUMN_LEVENSHTEIN_RATIO].transform(max) == remaining[c.COLUMN_LEVENSHTEIN_RATIO]
        remaining = remaining.loc[indexes_with_max_ratios, :]
        remaining = remaining.loc[remaining[c.COLUMN_LEVENSHTEIN_RATIO] > 94, :]

        for test_index, title_to_match, best_match, best_match_id in zip(
            remaining[c.COLUMN_TEST_INDEX],
            remaining[c.COLUMN_TRANSFORMED_TITLE],
            remaining[c.COLUMN_MATCH_TRANSFORMED_TITLE],
            remaining[c.COLUMN_MATCH_TITLE_ID],
        ):
            self._save_prediction(test_index, title_to_match, best_match, best_match_id, 1.0)

        del remaining

    def _find_matches_using_model(self):
        LOGGER.info('Finding matches using the model!')

        remaining = self._combine_titles_with_matches(100)
        number_of_rows_remaining = len(remaining)

        prediction_features = np.zeros((number_of_rows_remaining,), dtype=s.FEATURES_TYPES)
        for matrix_index, (title, truth_title_to_match_with) in enumerate(
                zip(remaining[c.COLUMN_TRANSFORMED_TITLE], remaining[c.COLUMN_MATCH_TRANSFORMED_TITLE])):

            if not((matrix_index+1) % 10000):
                print(f'Processed {matrix_index+1} of {number_of_rows_remaining}!')

            kind, target = s.DISABLE_TRAIN_KIND_VALUE, s.DISABLE_TARGET_VALUE
            prediction_features[matrix_index] = self.feature_engineering.construct_features(
                kind, title, truth_title_to_match_with, target)

        remaining.loc[:, c.COLUMN_PREDICTION] = self._predict(prediction_features)

        indexes_with_max_predictions = remaining.groupby(
            [c.COLUMN_TEST_INDEX])[c.COLUMN_PREDICTION].transform(max) == remaining[c.COLUMN_PREDICTION]
        remaining = remaining.loc[indexes_with_max_predictions, :]
        remaining = remaining.loc[remaining[c.COLUMN_PREDICTION] > s.PREDICTION_PROBABILITY_THRESHOLD, :]

        for test_index, title_to_match, best_match, best_match_id, prediction_probability in zip(
                remaining[c.COLUMN_TEST_INDEX],
                remaining[c.COLUMN_TRANSFORMED_TITLE],
                remaining[c.COLUMN_MATCH_TRANSFORMED_TITLE],
                remaining[c.COLUMN_MATCH_TITLE_ID],
                remaining[c.COLUMN_PREDICTION],
        ):
            self._save_prediction(test_index, title_to_match, best_match, best_match_id, prediction_probability)

        del remaining

    def _update_matched_so_far(self):
        filter_ = self.predictions[c.COLUMN_BEST_MATCH_ID] != s.TRAIN_NOT_FOUND_VALUE
        self.matched_so_far = list(self.predictions.loc[filter_, :].index)
        LOGGER.info(f'Matched {len(self.matched_so_far)} titles so far!')

    def _finalize_output(self):
        predictions = self.predictions.loc[:, [c.COLUMN_BEST_MATCH_ID]].copy(deep=True)
        predictions.loc[:, c.COLUMN_TEST_INDEX] = predictions.index
        predictions.rename(columns={c.COLUMN_BEST_MATCH_ID: c.COLUMN_TITLE_ID}, inplace=True)
        predictions.to_csv(s.FINAL_OUTPUT_FILE, index=False, sep=s.TEST_FILE_DELIMITER)

    def extensive_search_single_title(self, title):
        title_to_match = transform_title(title)
        matches_nearest = list(self.truth_data.index)

        prediction_features = np.zeros((len(matches_nearest),), dtype=s.FEATURES_TYPES)
        matches = []
        for matrix_index, match_index in enumerate(matches_nearest):
            match = self.truth_data_mapping[match_index]
            matches.append(match)

            kind, title, truth_title, target = (
                s.DISABLE_TRAIN_KIND_VALUE, title_to_match, match, s.DISABLE_TARGET_VALUE
            )
            prediction_features[matrix_index] = self.feature_engineering.construct_features(
                kind, title, truth_title, target)

        predictions = self._predict(prediction_features)

        best_match_index = np.argmax(predictions)
        best_match = matches[best_match_index]
        best_match_id = matches_nearest[best_match_index]
        best_match_prediction = predictions[best_match_index]

        return best_match_id, best_match, best_match_prediction

    def _find_close_matches_ratio(self):
        return self._find_close_matches(False)

    def _find_close_matches_token_sort_ratio(self):
        return self._find_close_matches(True)

    def process(self):
        steps = [
            self._find_exact_matches,
            self._find_close_matches_ratio,
            self._find_close_matches_token_sort_ratio,
            self._find_matches_using_model
        ]

        for step in steps:
            step()
            self._update_matched_so_far()

        self._finalize_output()

        return s.FINAL_OUTPUT_FILE
