import logging
import _pickle as pickle

import pandas as pd
import numpy as np
import xgboost as xgb

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import levenshtein_ratio, levenshtein_token_sort_ratio
from doppelspeller.feature_engineering import FEATURES_COUNT, FeatureEngineering, construct_features
from doppelspeller.match_maker import MatchMaker

LOGGER = logging.getLogger(__name__)


class Prediction:
    def __init__(self, data_type, title=None):
        self.feature_engineering = FeatureEngineering(data_type, title=title)
        self.data = self.feature_engineering.data
        self.truth_data = self.feature_engineering.truth_data
        self.model = self._load_model()

        self.predictions = None
        self.match_maker = None
        self.matched_so_far = None
        self.truth_data_mapping = None
        self.truth_data_mapping_reversed = None

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
        """
        TODO: THIS IS SLOW
        :return:
        """
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

        self._update_matched_so_far()

    def _get_nearest_match_title_id(self, index, test_index, nearest_matches):
        index = index % self.match_maker.top_n
        return nearest_matches[test_index][index]

    def _get_nearest_match_title(self, match_id):
        return self.truth_data_mapping[match_id]

    def _combine_titles_with_matches(self):
        remaining = self.data.loc[~self.data.index.isin(self.matched_so_far), :].copy(deep=True)
        remaining = remaining.loc[:, [c.COLUMN_TEST_INDEX, c.COLUMN_TRANSFORMED_TITLE]]
        nearest_matches = {row_number: self.match_maker.get_closest_matches(row_number)
                           for row_number in remaining.index}

        remaining = remaining.loc[remaining.index.repeat(self.match_maker.top_n)]
        remaining.reset_index(drop=True, inplace=True)

        matches_title_ids = map(lambda x, y: self._get_nearest_match_title_id(x, y, nearest_matches),
                                remaining.index, remaining[c.COLUMN_TEST_INDEX])
        remaining.loc[:, c.COLUMN_MATCH_TITLE_ID] = list(matches_title_ids)
        remaining.loc[:, c.COLUMN_MATCH_TRANSFORMED_TITLE] = remaining[c.COLUMN_MATCH_TITLE_ID].apply(
            self._get_nearest_match_title)

        return remaining

    @staticmethod
    def _get_levenshtein_ratio(x, y, threshold):
        ratio = levenshtein_ratio(x, y)
        if ratio <= threshold:
            return levenshtein_token_sort_ratio(x, y)
        return ratio

    def _find_close_matches(self):
        """
        TODO: THIS IS SLOW
        """
        threshold = 94

        LOGGER.info(f'Finding very close matches!')

        remaining = self._combine_titles_with_matches()

        matches_ratios = map(lambda x, y: self._get_levenshtein_ratio(x, y, threshold),
                             remaining[c.COLUMN_TRANSFORMED_TITLE], remaining[c.COLUMN_MATCH_TRANSFORMED_TITLE])
        remaining.loc[:, c.COLUMN_LEVENSHTEIN_RATIO] = list(matches_ratios)

        indexes_with_max_ratios = remaining.groupby(
            [c.COLUMN_TEST_INDEX])[c.COLUMN_LEVENSHTEIN_RATIO].transform(max) == remaining[c.COLUMN_LEVENSHTEIN_RATIO]

        matches = remaining.loc[indexes_with_max_ratios, :]
        matches = matches.loc[matches[c.COLUMN_LEVENSHTEIN_RATIO] > threshold, :]
        for test_index, title_to_match, best_match, best_match_id in zip(
            matches[c.COLUMN_TEST_INDEX],
            matches[c.COLUMN_TRANSFORMED_TITLE],
            matches[c.COLUMN_MATCH_TRANSFORMED_TITLE],
            matches[c.COLUMN_MATCH_TITLE_ID],
        ):
            self._save_prediction(test_index, title_to_match, best_match, best_match_id, 1.0)

        self._update_matched_so_far()

        found_test_indexes = list(matches[c.COLUMN_TEST_INDEX].unique())
        return remaining.loc[~(remaining[c.COLUMN_TEST_INDEX].isin(found_test_indexes)), :]

    def _find_matches_using_model(self, remaining, single_prediction=False):
        LOGGER.info('Finding matches using the model!')

        number_of_rows = len(remaining)

        # TODO - THIS IS SLOW
        ########################################################################################################
        LOGGER.info('Encoding data for constructing the features!')

        encoding_type = s.NUMBER_OF_CHARACTERS_DATA_TYPE
        float_type = s.ENCODING_FLOAT_TYPE

        title_number_of_characters = np.array(remaining[c.COLUMN_TRANSFORMED_TITLE].str.len(), dtype=encoding_type)
        truth_number_of_characters = np.array(
            remaining[c.COLUMN_MATCH_TRANSFORMED_TITLE].str.len(), dtype=encoding_type)
        title_encoded = np.array(remaining[c.COLUMN_TRANSFORMED_TITLE].apply(
            self.feature_engineering.encode_title).tolist(), dtype=encoding_type)
        title_truth_encoded = np.array(remaining[c.COLUMN_MATCH_TRANSFORMED_TITLE].apply(
            self.feature_engineering.encode_title).tolist(), dtype=encoding_type)
        word_counter_encoded = np.array(remaining[c.COLUMN_MATCH_TRANSFORMED_TITLE].apply(
            self.feature_engineering.encode_word_counter).tolist(), dtype=encoding_type)

        LOGGER.info('Data encoded!')
        ########################################################################################################

        LOGGER.info(f'Constructing features!')

        features = np.zeros((number_of_rows, FEATURES_COUNT), dtype=float_type)
        dummy = np.zeros((FEATURES_COUNT,), dtype=encoding_type)
        construct_features(title_number_of_characters, truth_number_of_characters,
                           title_encoded, title_truth_encoded, word_counter_encoded,
                           self.feature_engineering.space_code, self.feature_engineering.number_of_truth_titles,
                           dummy, features)

        LOGGER.info(f'Features (shape = {features.shape}) constructed!')

        LOGGER.info('Calling model.predict()!')
        remaining.loc[:, c.COLUMN_PREDICTION] = self.model.predict(xgb.DMatrix(features))
        LOGGER.info('Predictions generated!')

        LOGGER.info('Saving predictions!')

        if single_prediction:
            max_prediction = max(remaining.loc[:, c.COLUMN_PREDICTION])
            matches = remaining.loc[remaining[c.COLUMN_PREDICTION] == max_prediction, :]
        else:
            indexes_with_max_predictions = remaining.groupby(
                [c.COLUMN_TEST_INDEX])[c.COLUMN_PREDICTION].transform(max) == remaining[c.COLUMN_PREDICTION]
            matches = remaining.loc[indexes_with_max_predictions, :]
            matches = matches.loc[matches[c.COLUMN_PREDICTION] > s.PREDICTION_PROBABILITY_THRESHOLD, :]

        for test_index, title_to_match, best_match, best_match_id, prediction_probability in zip(
                matches[c.COLUMN_TEST_INDEX],
                matches[c.COLUMN_TRANSFORMED_TITLE],
                matches[c.COLUMN_MATCH_TRANSFORMED_TITLE],
                matches[c.COLUMN_MATCH_TITLE_ID],
                matches[c.COLUMN_PREDICTION],
        ):
            self._save_prediction(test_index, title_to_match, best_match, best_match_id, prediction_probability)

        del remaining

        self._update_matched_so_far()

        LOGGER.info('Predictions saved!')

    def _update_matched_so_far(self):
        filter_ = self.predictions[c.COLUMN_BEST_MATCH_ID] != s.TRAIN_NOT_FOUND_VALUE
        self.matched_so_far = list(self.predictions.loc[filter_, :].index)
        LOGGER.info(f'Matched {len(self.matched_so_far)} titles so far!')

    def _finalize_output(self):
        predictions = self.predictions.loc[:, [c.COLUMN_BEST_MATCH_ID]].copy(deep=True)
        predictions.loc[:, c.COLUMN_TEST_INDEX] = predictions.index
        predictions.rename(columns={c.COLUMN_BEST_MATCH_ID: c.COLUMN_TITLE_ID}, inplace=True)
        predictions.to_csv(s.FINAL_OUTPUT_FILE, index=False, sep=s.TEST_FILE_DELIMITER)

    def generate_test_predictions(self, single_prediction=False):
        top_n = s.TOP_N_RESULTS_TO_FIND_FOR_PREDICTING
        if single_prediction:
            top_n = self.feature_engineering.number_of_truth_titles
        self.matched_so_far = []
        self.match_maker = MatchMaker(self.data, self.truth_data, top_n)
        self.predictions = self._initiate_predictions_data(list(self.data[c.COLUMN_TEST_INDEX]))
        self.feature_engineering = FeatureEngineering(c.DATA_TYPE_TEST)
        self.truth_data_mapping, self.truth_data_mapping_reversed = self._get_truth_data_mappings(self.truth_data)

        self._find_exact_matches()

        chunk_size = 5000
        data = self.data.copy(deep=True)
        total = len(data)
        iteration = -1
        while True:
            iteration += 1
            start_index = iteration * chunk_size
            stop_index = start_index + chunk_size

            self.data = data.loc[data.iloc[start_index:stop_index].index, :]
            if self.data.empty:
                break

            if stop_index > total:
                stop_index = total
            LOGGER.info(f'\nProcessing {start_index}-{stop_index} of {total}!\n')

            remaining = self._find_close_matches()
            self._find_matches_using_model(remaining, single_prediction=single_prediction)

        if single_prediction:
            return self.predictions.iloc[0].to_dict()

        self._finalize_output()
        return s.FINAL_OUTPUT_FILE
