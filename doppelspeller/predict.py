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
    """
    Class responsible for generating predictions, given a "data_type" or a single "title"

    :param data_type: See DATA_TYPE_MAPPING in doppelspeller.feature_engineering
    :param title: Must be provided if data_type == c.DATA_TYPE_SINGLE

    * Main public method: generate_test_predictions(...)
    """
    def __init__(self, data_type, title=None):
        self.fe = FeatureEngineering(data_type, title=title)
        self.test_indexes = list(self.fe.data[c.COLUMN_TEST_INDEX])
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

        # Encoding/caching mapping to speed up features generation
        self.mapping_truth_title_encoding = None
        self.mapping_truth_words_counts = None
        self.mapping_title_encoding = None
        self._populate_encoding_mappings()

        self.data = None

    def _populate_encoding_mappings(self):
        title_column = c.COLUMN_TRANSFORMED_TITLE

        self.mapping_truth_title_encoding = self.fe.truth_data.set_index(c.COLUMN_TITLE_ID).to_dict()[title_column]
        self.mapping_truth_title_encoding = {
            k: self.fe.encode_title(v) for k, v in self.mapping_truth_title_encoding.items()
        }

        self.mapping_truth_words_counts = self.fe.truth_data.set_index(c.COLUMN_TITLE_ID).to_dict()[title_column]
        self.mapping_truth_words_counts = {
            k: self.fe.get_truth_words_counts(v) for k, v in self.mapping_truth_words_counts.items()
        }

        self.mapping_title_encoding = self.fe.data.set_index(c.COLUMN_TEST_INDEX).to_dict()[title_column]
        self.mapping_title_encoding = {
            k: self.fe.encode_title(v) for k, v in self.mapping_title_encoding.items()
        }

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

        # Change data types
        self.predictions.loc[:, c.COLUMN_TEST_INDEX] = self.predictions[c.COLUMN_TEST_INDEX].astype(np.uint32)
        self.predictions.loc[:, c.COLUMN_MATCH_TITLE_ID] = self.predictions[c.COLUMN_MATCH_TITLE_ID].astype(np.uint32)
        self.predictions.loc[:, c.COLUMN_PREDICTION] = self.predictions[c.COLUMN_PREDICTION].astype(np.float16)

        self.matched_so_far = list(self.predictions[c.COLUMN_TEST_INDEX])

        LOGGER.info(f'Matched {len(self.matched_so_far)} titles so far!')

    def _find_exact_matches(self):
        LOGGER.info('Finding exact matches!')

        exact_value_flag = -2
        self.fe.data.loc[:, c.COLUMN_EXACT] = self.fe.data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
            lambda x: self.truth_data_mapping_reversed.get(x, exact_value_flag))

        del self.truth_data_mapping_reversed

        test_data_filtered = self.fe.data.loc[self.fe.data[c.COLUMN_EXACT] != exact_value_flag, :].copy(deep=True)
        if not test_data_filtered.empty:
            test_data_filtered.loc[:, c.COLUMN_PREDICTION] = 1.0
            test_data_filtered.loc[:, c.COLUMN_MATCH_TRANSFORMED_TITLE] = test_data_filtered.loc[
                                                                          :, c.COLUMN_TRANSFORMED_TITLE]
            test_data_filtered.rename(columns={c.COLUMN_EXACT: c.COLUMN_MATCH_TITLE_ID}, inplace=True)

            self._save_prediction(test_data_filtered)

    def _get_nearest_match_title_id(self, index, test_index, nearest_matches):
        index = index % self.match_maker.top_n
        return nearest_matches[test_index][index]

    def _get_nearest_match_title(self, match_id):
        return self.truth_data_mapping[match_id]

    def _combine_titles_with_matches(self):
        remaining = self.data.loc[~self.data.index.isin(self.matched_so_far),
                                  [c.COLUMN_TEST_INDEX, c.COLUMN_TRANSFORMED_TITLE]].copy(deep=True)

        nearest_matches = {row_number: self.match_maker.get_closest_matches(row_number)
                           for row_number in remaining.index}

        remaining = remaining.reindex(remaining.index.repeat(self.match_maker.top_n))
        remaining.reset_index(drop=True, inplace=True)

        matches_title_ids = map(lambda x, y: self._get_nearest_match_title_id(x, y, nearest_matches),
                                remaining.index, remaining[c.COLUMN_TEST_INDEX])
        remaining.loc[:, c.COLUMN_MATCH_TITLE_ID] = list(matches_title_ids)
        remaining.loc[:, c.COLUMN_MATCH_TRANSFORMED_TITLE] = remaining[c.COLUMN_MATCH_TITLE_ID].apply(
            self._get_nearest_match_title)

        return remaining

    @staticmethod
    def _get_levenshtein_deletion_ratio(x, y):
        length_x, length_y = len(x), len(y)
        total_length = length_x + length_y
        delta = abs(length_x - length_y)
        return ((total_length - delta) / total_length) * 100

    @classmethod
    def _get_levenshtein_ratio(cls, x, y):
        # We only consider the levenshtein_ratio for "matching" if it's > threshold
        if cls._get_levenshtein_deletion_ratio(x, y) < s.LEVENSHTEIN_RATIO_THRESHOLD:
            return 0

        ratio = levenshtein_ratio(x, y)
        if ratio <= s.LEVENSHTEIN_RATIO_THRESHOLD:
            return levenshtein_token_sort_ratio(x, y)
        return ratio

    @staticmethod
    def _remove_duplicated_matches(matches):
        duplicated_matches = matches.loc[matches.duplicated([c.COLUMN_TEST_INDEX]), c.COLUMN_TEST_INDEX]
        return matches.loc[~(matches[c.COLUMN_TEST_INDEX].isin(duplicated_matches)), :]

    def _find_close_matches(self):
        LOGGER.info(f'Finding very close matches!')

        remaining = self._combine_titles_with_matches()

        matches_ratios = map(lambda x, y: self._get_levenshtein_ratio(x, y),
                             remaining[c.COLUMN_TRANSFORMED_TITLE], remaining[c.COLUMN_MATCH_TRANSFORMED_TITLE])
        remaining.loc[:, c.COLUMN_LEVENSHTEIN_RATIO] = list(matches_ratios)

        matches = remaining.loc[remaining[c.COLUMN_LEVENSHTEIN_RATIO] > s.LEVENSHTEIN_RATIO_THRESHOLD, :]
        indexes_with_max_ratios = matches.groupby(
            [c.COLUMN_TEST_INDEX])[c.COLUMN_LEVENSHTEIN_RATIO].transform(max) == matches[c.COLUMN_LEVENSHTEIN_RATIO]

        matches = self._remove_duplicated_matches(matches.loc[indexes_with_max_ratios, :])

        if not matches.empty:
            matches.loc[:, c.COLUMN_PREDICTION] = 1.0

            self._save_prediction(matches)

        return remaining.loc[~(remaining[c.COLUMN_TEST_INDEX].isin(self.matched_so_far)), :]

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

        # http://numba.pydata.org/numba-doc/latest/reference/fpsemantics.html#warnings-and-errors
        # Ignoring an invalid warning, as it can not be reproduced with forceobj=True
        with np.errstate(all='ignore'):
            construct_features(title_number_of_characters, truth_number_of_characters,
                               title_encoded, title_truth_encoded, truth_words_counts,
                               self.fe.space_code, self.fe.number_of_truth_titles,
                               dummy, features)

        LOGGER.info(f'Features (shape = {features.shape}) constructed!')

        del title_number_of_characters
        del truth_number_of_characters
        del title_encoded
        del title_truth_encoded
        del truth_words_counts

        features_d = xgb.DMatrix(features)
        del features

        LOGGER.info('Calling model.predict()!')
        # TODO: model.predict(...) seems to be slow
        remaining.loc[:, c.COLUMN_PREDICTION] = self.model.predict(features_d, ntree_limit=self.model.best_ntree_limit)
        LOGGER.info('Predictions generated!')

        LOGGER.info('Saving predictions!')

        if single_prediction:
            max_prediction = max(remaining.loc[:, c.COLUMN_PREDICTION])
            matches = remaining.loc[remaining[c.COLUMN_PREDICTION] == max_prediction, :].head(1)
            self._save_prediction(matches)
        else:
            indexes_with_max_predictions = remaining.groupby(
                [c.COLUMN_TEST_INDEX])[c.COLUMN_PREDICTION].transform(max) == remaining[c.COLUMN_PREDICTION]
            matches = remaining.loc[indexes_with_max_predictions, :]
            matches = self._remove_duplicated_matches(
                matches.loc[matches[c.COLUMN_PREDICTION] > s.PREDICTION_PROBABILITY_THRESHOLD, :]
            )

            if not matches.empty:
                self._save_prediction(matches)

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

        LOGGER.info(f'\n\n{"*" * 100}\nOutput saved to {s.FINAL_OUTPUT_FILE}\n{"*" * 100}\n')

    def generate_test_predictions(self, single_prediction=False):
        """
        * Gets exact title matches
        * Gets all the nearest titles, for self.data, using self.match_maker
        * For the nearest titles, first tries to get all the matches using self._get_levenshtein_ratio(...)
        * Then the remaining titles are matched using the trained model
        * The results are finalized and saved using self._finalize_output()
        """
        if single_prediction:
            if len(self.fe.data) != 1:
                raise Exception(f'For "single_prediction" len(self.data) should be 1 (is {len(self.data)})!')

        top_n = s.TOP_N_RESULTS_TO_FIND_FOR_PREDICTING

        self.matched_so_far = []
        self.match_maker = MatchMaker(self.fe.data, self.fe.truth_data, top_n)
        self.truth_data_mapping, self.truth_data_mapping_reversed = self._get_truth_data_mappings(self.fe.truth_data)

        self._find_exact_matches()

        chunk_size = 10000
        data = self.fe.data.loc[:, [c.COLUMN_TEST_INDEX, c.COLUMN_TRANSFORMED_TITLE]].copy(deep=True)
        data.loc[:, c.COLUMN_TEST_INDEX] = data[c.COLUMN_TEST_INDEX].astype(np.uint32)
        total = len(data)
        iteration = -1
        while True:
            iteration += 1
            start_index = iteration * chunk_size
            stop_index = start_index + chunk_size

            self.data = data.loc[data.iloc[start_index:stop_index].index,
                                 [c.COLUMN_TEST_INDEX, c.COLUMN_TRANSFORMED_TITLE]]
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
