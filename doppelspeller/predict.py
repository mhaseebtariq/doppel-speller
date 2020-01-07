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
        self.test_indexes = list(self.data[c.COLUMN_TEST_INDEX])
        self.truth_data = self.feature_engineering.truth_data
        self.model = self._load_model()

        self.match_maker = None
        self.matched_so_far = None
        self.truth_data_mapping = None
        self.truth_data_mapping_reversed = None
        self.predictions_columns = [
            c.COLUMN_TEST_INDEX,
            c.COLUMN_TRANSFORMED_TITLE,
            c.COLUMN_MATCH_TRANSFORMED_TITLE,
            c.COLUMN_MATCH_TITLE_ID,
            c.COLUMN_PREDICTION,
        ]
        self.predictions = pd.DataFrame(index=[], columns=self.predictions_columns)

        self.mapping_truth_title_encoding = None
        self.mapping_truth_words_counts = None
        self.mapping_title_encoding = None
        self._populate_encoding_mappings()

    def _populate_encoding_mappings(self):
        title_column = c.COLUMN_TRANSFORMED_TITLE

        self.mapping_truth_title_encoding = self.truth_data.set_index(c.COLUMN_TITLE_ID).to_dict()[title_column]
        self.mapping_truth_title_encoding = {
            k: self.feature_engineering.encode_title(v) for k, v in self.mapping_truth_title_encoding.items()
        }
        self.mapping_truth_words_counts = self.truth_data.set_index(c.COLUMN_TITLE_ID).to_dict()[title_column]
        self.mapping_truth_words_counts = {
            k: self.feature_engineering.get_truth_words_counts(v) for k, v in self.mapping_truth_words_counts.items()
        }
        self.mapping_title_encoding = self.data.set_index(c.COLUMN_TEST_INDEX).to_dict()[title_column]
        self.mapping_title_encoding = {
            k: self.feature_engineering.encode_title(v) for k, v in self.mapping_title_encoding.items()
        }

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

    def _save_prediction(self, matches):
        self.predictions = pd.concat([self.predictions, matches.loc[:, self.predictions_columns]],
                                     axis=0, ignore_index=True)
        self.matched_so_far = list(self.predictions[c.COLUMN_TEST_INDEX])

        LOGGER.info(f'Matched {len(self.matched_so_far)} titles so far!')

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

        test_data_filtered = self.data.loc[self.data[c.COLUMN_EXACT] != exact_value_flag, :].copy(deep=True)
        if not test_data_filtered.empty:
            test_data_filtered.loc[:, c.COLUMN_PREDICTION] = 1.0
            test_data_filtered.loc[:, c.COLUMN_MATCH_TRANSFORMED_TITLE] = test_data_filtered.loc[
                                                                          :, c.COLUMN_TRANSFORMED_TITLE]
            test_data_filtered.rename(columns={c.COLUMN_EXACT: c.COLUMN_MATCH_TITLE_ID}, inplace=True)

            self._save_prediction(test_data_filtered)

        del test_data_filtered

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
        if not matches.empty:
            matches.loc[:, c.COLUMN_PREDICTION] = 1.0

            self._save_prediction(matches)

        return remaining.loc[~(remaining[c.COLUMN_TEST_INDEX].isin(matches[c.COLUMN_TEST_INDEX].tolist())), :]

    def _find_matches_using_model(self, remaining, single_prediction=False):
        LOGGER.info('Finding matches using the model!')

        number_of_rows = len(remaining)

        LOGGER.info('Encoding data for constructing the features!')

        encoding_type = s.NUMBER_OF_CHARACTERS_DATA_TYPE
        float_type = s.ENCODING_FLOAT_TYPE

        title_number_of_characters = np.array(remaining[c.COLUMN_TRANSFORMED_TITLE].str.len(), dtype=encoding_type)
        truth_number_of_characters = np.array(
            remaining[c.COLUMN_MATCH_TRANSFORMED_TITLE].str.len(), dtype=encoding_type)

        title_encoded = np.vstack(
            remaining[c.COLUMN_TEST_INDEX].apply(lambda x: self.mapping_title_encoding[x]).tolist())
        title_truth_encoded = np.vstack(
            remaining[c.COLUMN_MATCH_TITLE_ID].apply(lambda x: self.mapping_truth_title_encoding[x]).tolist())
        truth_words_counts = np.vstack(
            remaining[c.COLUMN_MATCH_TITLE_ID].apply(lambda x: self.mapping_truth_words_counts[x]).tolist())

        LOGGER.info('Data encoded!')

        LOGGER.info(f'Constructing features!')

        features = np.zeros((number_of_rows, FEATURES_COUNT), dtype=float_type)
        dummy = np.zeros((FEATURES_COUNT,), dtype=encoding_type)
        construct_features(title_number_of_characters, truth_number_of_characters,
                           title_encoded, title_truth_encoded, truth_words_counts,
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
            self._save_prediction(matches)
        else:
            indexes_with_max_predictions = remaining.groupby(
                [c.COLUMN_TEST_INDEX])[c.COLUMN_PREDICTION].transform(max) == remaining[c.COLUMN_PREDICTION]
            matches = remaining.loc[indexes_with_max_predictions, :]
            matches = matches.loc[matches[c.COLUMN_PREDICTION] > s.PREDICTION_PROBABILITY_THRESHOLD, :]
            if not matches.empty:
                self._save_prediction(matches)

        del remaining

        LOGGER.info('Predictions saved!')

    def _finalize_output(self):
        LOGGER.info('Finalizing output!')

        predictions = self.predictions.loc[:, [c.COLUMN_MATCH_TITLE_ID, c.COLUMN_TEST_INDEX]].copy(deep=True)
        predictions.rename(columns={c.COLUMN_MATCH_TITLE_ID: c.COLUMN_TITLE_ID}, inplace=True)

        not_found_indexes = list(set(self.test_indexes).difference(predictions[c.COLUMN_TEST_INDEX]))
        if not_found_indexes:
            not_found = pd.DataFrame(not_found_indexes, columns=[c.COLUMN_TEST_INDEX])
            not_found.loc[:, c.COLUMN_TITLE_ID] = s.TRAIN_NOT_FOUND_VALUE
            not_found = not_found.loc[:, [c.COLUMN_TITLE_ID, c.COLUMN_TEST_INDEX]]
            predictions = pd.concat([predictions, not_found], axis=0, ignore_index=True)

        predictions.sort_values(c.COLUMN_TEST_INDEX, inplace=True)
        predictions.to_csv(s.FINAL_OUTPUT_FILE, index=False, sep=s.TEST_FILE_DELIMITER)

        LOGGER.info('Finalized output!')

    def generate_test_predictions(self, single_prediction=False):
        top_n = s.TOP_N_RESULTS_TO_FIND_FOR_PREDICTING

        self.matched_so_far = []
        self.match_maker = MatchMaker(self.data, self.truth_data, top_n)
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
