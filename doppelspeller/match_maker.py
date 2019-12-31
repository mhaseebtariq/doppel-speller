import math
import time
import logging

import numpy as np
from scipy.sparse import lil_matrix

import doppelspeller.constants as c
import doppelspeller.settings as s
from doppelspeller import njit, List
from doppelspeller.common import get_n_grams_counter, get_ground_truth, get_train_data, get_test_data

NP_ACTUAL_ERROR_CONFIG = np.geterr()
ENCODING_FLOAT_TYPE = np.float32
DATA_TYPE_MAPPING = {
    c.DATA_TYPE_TRAIN: get_train_data,
    c.DATA_TYPE_TEST: get_test_data,
}
NEAREST_N_MAPPING = {
    c.DATA_TYPE_TRAIN: s.TOP_N_RESULTS_TO_FIND_FOR_TRAINING,
    c.DATA_TYPE_TEST: s.TOP_N_RESULTS_TO_FIND_FOR_PREDICTING,
}


LOGGER = logging.getLogger(__name__)


@njit(fastmath=True, parallel=True)
def fast_jaccard(number_of_truth_titles, max_intersection_possible, non_zero_columns_for_the_row,
                 matrix_truth_non_zero_columns_and_values, sums_matrix_truth):
    """
    TODO: Set the proper function signatures for better speed!
    """
    scores = np.zeros((number_of_truth_titles,), dtype=ENCODING_FLOAT_TYPE)
    for non_zero_column in non_zero_columns_for_the_row:
        columns, values = matrix_truth_non_zero_columns_and_values[non_zero_column]
        scores[columns] += values

    return -(scores / (sums_matrix_truth + (max_intersection_possible - scores)))


class MatchMaker:
    data = None
    truth_data = None
    n_grams_counter = None
    n_grams_counter_truth = None
    number_of_truth_titles = None
    idf_s_mapping = None
    n_grams_decoding = None
    n_grams_encoding = None
    matrix = None
    matrix_truth = None
    sums_matrix_truth = None
    closest_matches = []
    matrix_truth_non_zero_columns_and_values = List()
    matrix_non_zero_columns = List()

    def __init__(self, data_type, approximate=False):
        self.data_type = data_type
        self.approximate = approximate

        self.top_n = NEAREST_N_MAPPING[self.data_type]

    def _idf(self, word):
        """
        Inverse document frequency
        """
        return math.log(self.number_of_truth_titles / self.n_grams_counter_truth[word])

    def _populate_idf_s_mapping(self):
        self.idf_s_mapping = {key: self._idf(key) for key in self.n_grams_counter_truth.keys()}
        self.max_idf_value = max(self.idf_s_mapping.values())

    def _populate_encoding_mappings(self):
        all_n_grams = set(list(self.n_grams_counter.keys()) + list(self.n_grams_counter_truth.keys()))

        self.n_grams_decoding = {index: ngram for index, ngram in enumerate(all_n_grams)}
        self.n_grams_encoding = {v: k for k, v in self.n_grams_decoding.items()}

    def _get_encoding_values(self, value):
        indexes = [self.n_grams_encoding[x] for x in value]
        uniqueness_values = np.array([self.idf_s_mapping.get(x, self.max_idf_value) for x in value],
                                     dtype=ENCODING_FLOAT_TYPE)
        return indexes, uniqueness_values

    def _construct_sparse_matrix(self, data):
        matrix = lil_matrix((len(data), len(self.n_grams_encoding)), dtype=ENCODING_FLOAT_TYPE)
        for index, value in enumerate(data[c.COLUMN_N_GRAMS]):
            indexes, uniqueness_values = self._get_encoding_values(value)
            matrix[index, indexes] = uniqueness_values
        return matrix

    def _get_idf_given_index(self, index):
        return self.idf_s_mapping.get(self.n_grams_decoding[index], self.max_idf_value)

    @staticmethod
    def _top_k_hybrid(array, k, axis=-1):
        """
        Taken from : https://stackoverflow.com/a/42186357
        """
        arg_partition = array.argpartition(k, axis=axis)[(*axis % array.ndim * (slice(None),), slice(k))]
        return np.take_along_axis(arg_partition, np.take_along_axis(array, arg_partition, axis).argsort(axis), axis)

    def _get_top_n_matches(self, modified_jaccard):
        if self.approximate:
            modified_jaccard = np.array(modified_jaccard, dtype=np.float16)

        # top_matches = modified_jaccard.argpartition(range(self.top_n))[:self.top_n]
        # top_matches = modified_jaccard.argsort()[:self.top_n]
        top_matches = self._top_k_hybrid(modified_jaccard, self.top_n)
        return np.array(self.truth_data.loc[top_matches, c.COLUMN_TITLE_ID].values, dtype=np.uint32)

    def _find_matches(self):
        np.seterr(divide='ignore', invalid='ignore')

        # Delete all unwanted variables
        del self.n_grams_counter
        del self.n_grams_counter_truth
        del self.n_grams_encoding
        del self.matrix
        del self.matrix_truth

        iteration_start = time.time()
        total = len(self.matrix_non_zero_columns)
        for count, non_zero_columns_for_the_row in enumerate(self.matrix_non_zero_columns):
            if not (count + 1) % 5000:
                LOGGER.info(f'Processed: {count + 1} of {total} | '
                            f'Iteration time: {round(time.time() - iteration_start, 2)}')
                iteration_start = time.time()

            max_intersection_possible = sum([self._get_idf_given_index(r) for r in non_zero_columns_for_the_row])

            modified_jaccard = fast_jaccard(
                self.number_of_truth_titles, max_intersection_possible, non_zero_columns_for_the_row,
                self.matrix_truth_non_zero_columns_and_values, self.sums_matrix_truth
            )

            self.closest_matches.append(self._get_top_n_matches(modified_jaccard))

        _ = np.seterr(**NP_ACTUAL_ERROR_CONFIG)

    def process(self):
        self.data = DATA_TYPE_MAPPING[self.data_type](trim_large_titles=False)
        self.truth_data = get_ground_truth(trim_large_titles=False)
        self.number_of_truth_titles = len(self.truth_data)

        self.n_grams_counter = get_n_grams_counter(self.data)
        self.n_grams_counter_truth = get_n_grams_counter(self.truth_data)

        self._populate_idf_s_mapping()
        self._populate_encoding_mappings()

        self.matrix = self._construct_sparse_matrix(self.data)
        self.matrix_truth = self._construct_sparse_matrix(self.truth_data)
        self.matrix_truth = lil_matrix(self.matrix_truth.T, dtype=ENCODING_FLOAT_TYPE)

        for row in range(self.matrix.shape[0]):
            self.matrix_non_zero_columns.append(self.matrix[row].nonzero()[1])

        for row in range(self.matrix_truth.shape[0]):
            non_zero_columns = self.matrix_truth[row].nonzero()[1]
            values = np.array([self._get_idf_given_index(row)] * len(non_zero_columns), dtype=ENCODING_FLOAT_TYPE)
            self.matrix_truth_non_zero_columns_and_values.append((non_zero_columns, values))

        self.sums_matrix_truth = np.array(
            [x for x in np.sum(self.matrix_truth, axis=0).data.tolist()][0], dtype=ENCODING_FLOAT_TYPE
        )

        self._find_matches()
