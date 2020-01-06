import math
import logging

import numba
import numpy as np
from scipy.sparse import lil_matrix

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import get_n_grams_counter


LOGGER = logging.getLogger(__name__)


@numba.njit(fastmath=True, parallel=True)
def fast_jaccard(number_of_truth_titles, max_intersection_possible, non_zero_columns_for_the_row,
                 matrix_truth_non_zero_columns_and_values, sums_matrix_truth):
    """
    TODO
    """
    scores = np.zeros((number_of_truth_titles,), dtype=s.ENCODING_FLOAT_TYPE)
    for non_zero_column in non_zero_columns_for_the_row:
        columns, values = matrix_truth_non_zero_columns_and_values[non_zero_column]
        scores[columns] += values

    return scores / (sums_matrix_truth + (max_intersection_possible - scores))


@numba.njit(parallel=False)
def fast_top_k(array, k):
    """
    * 50x faster than np.argsort
    * Not sorted on importance
    """
    sorted_indexes = np.zeros((k,), dtype=s.ENCODING_FLOAT_TYPE)
    minimum_index = 0
    minimum_index_value = 0
    for value in array:
        if value > minimum_index_value:
            sorted_indexes[minimum_index] = value
            minimum_index = sorted_indexes.argmin()
            minimum_index_value = sorted_indexes[minimum_index]
    minimum_index_value -= s.ENCODING_FLOAT_BUFFER
    return (array >= minimum_index_value).nonzero()[0][::-1][:k]


class MatchMaker:
    def __init__(self, data, truth_data, top_n):
        self.data = data
        self.truth_data = truth_data
        self.top_n = top_n

        LOGGER.info(f'[{self.__class__.__name__}] Loading pre-requisite data!')

        self.n_grams_counter = get_n_grams_counter(self.data)
        self.n_grams_counter_truth = get_n_grams_counter(self.truth_data)
        self.number_of_truth_titles = len(self.truth_data)
        self.idf_s_mapping = self._get_idf_s_mapping()
        self.max_idf_value = max(self.idf_s_mapping.values())
        self.n_grams_decoding = self._get_encoding_mappings()
        self.n_grams_encoding = {v: k for k, v in self.n_grams_decoding.items()}

        self.matrix = self._construct_sparse_matrix(self.data)
        self.matrix_truth = self._construct_sparse_matrix(self.truth_data, transpose=True)
        self.matrix_non_zero_columns = self._get_matrix_non_zero_columns()
        self.matrix_truth_non_zero_columns_and_values = self._get_matrix_truth_non_zero_columns_and_values()
        self.sums_matrix_truth = self._get_sums_matrix_truth()

        LOGGER.info(f'[{self.__class__.__name__}] Loaded pre-requisite data!')

    def _get_matrix_non_zero_columns(self):
        matrix_non_zero_columns = numba.typed.List()
        for row in range(self.matrix.shape[0]):
            matrix_non_zero_columns.append(self.matrix[row].nonzero()[1])

        return matrix_non_zero_columns

    def _get_matrix_truth_non_zero_columns_and_values(self):
        matrix_truth_non_zero_columns_and_values = numba.typed.List()
        for row in range(self.matrix_truth.shape[0]):
            non_zero_columns = self.matrix_truth[row].nonzero()[1]
            values = np.array([self._get_idf_given_index(row)] * len(non_zero_columns), dtype=s.ENCODING_FLOAT_TYPE)
            matrix_truth_non_zero_columns_and_values.append((non_zero_columns, values))

        return matrix_truth_non_zero_columns_and_values

    def _get_sums_matrix_truth(self):
        return np.array(
            [x for x in np.sum(self.matrix_truth, axis=0).data.tolist()][0], dtype=s.ENCODING_FLOAT_TYPE
        )

    def _idf_n_gram(self, n_gram):
        """
        Inverse document frequency
        """
        return math.log(self.number_of_truth_titles / self.n_grams_counter_truth[n_gram])

    def _get_idf_s_mapping(self):
        return {key: self._idf_n_gram(key) for key in self.n_grams_counter_truth.keys()}

    def _get_encoding_mappings(self):
        all_n_grams = set(list(self.n_grams_counter.keys()) + list(self.n_grams_counter_truth.keys()))

        return {index: ngram for index, ngram in enumerate(all_n_grams)}

    def _get_encoding_values(self, value):
        indexes = [self.n_grams_encoding[x] for x in value]
        uniqueness_values = np.array([self.idf_s_mapping.get(x, self.max_idf_value) for x in value],
                                     dtype=s.ENCODING_FLOAT_TYPE)
        return indexes, uniqueness_values

    def _construct_sparse_matrix(self, data, transpose=False):
        LOGGER.info(f'[{self.__class__.__name__}] Constructing sparse matrix!')

        matrix = lil_matrix((len(data), len(self.n_grams_encoding)), dtype=s.ENCODING_FLOAT_TYPE)
        for index, value in enumerate(data[c.COLUMN_N_GRAMS]):
            indexes, uniqueness_values = self._get_encoding_values(value)
            matrix[index, indexes] = uniqueness_values

        if transpose:
            return lil_matrix(matrix.T, dtype=s.ENCODING_FLOAT_TYPE)

        LOGGER.info(f'[{self.__class__.__name__}] Constructed sparse matrix - {matrix.shape}!')

        return matrix

    def _get_idf_given_index(self, index):
        return self.idf_s_mapping.get(self.n_grams_decoding[index], self.max_idf_value)

    def _get_top_n_matches(self, modified_jaccard):
        top_matches = fast_top_k(modified_jaccard, self.top_n)
        if top_matches.shape[0] != self.top_n:
            raise Exception('top_matches.shape[0] != self.top_n')
        return self.truth_data.loc[top_matches, c.COLUMN_TITLE_ID].tolist()

    def get_closest_matches(self, row_number):
        non_zero_columns_for_the_row = self.matrix_non_zero_columns[row_number]
        max_intersection_possible = sum([self._get_idf_given_index(r) for r in non_zero_columns_for_the_row])

        modified_jaccard = fast_jaccard(
            self.number_of_truth_titles, max_intersection_possible, non_zero_columns_for_the_row,
            self.matrix_truth_non_zero_columns_and_values, self.sums_matrix_truth
        )
        return self._get_top_n_matches(modified_jaccard)
