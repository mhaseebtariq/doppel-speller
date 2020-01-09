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
    Calculates (modified) Jaccard distances using matrix operations.

    * The Jaccard distances are calculated for one title (data extracted from the arg -> non_zero_columns_for_the_row)
        - against the rest of all the titles in the "truth" database
    * In the returned array,
        - the 0th index is the Jaccard distance for the title and the 0th title in the truth database
        - the 1st index is the Jaccard distance for the title and the 1st title in the truth database
        - the 2nd index is the Jaccard distance for the title and the 2nd title in the truth database
        - and so on...
    * The "modified" version is explained as follows:
        - Instead of: (number of common n-grams in both titles) / (total number of unique n-grams in both titles)
        - The distance is calculated as:
            (Sum of IDFs of the common n-grams in both titles) / (Sum of IDFs of all the unique n-grams in both titles)
        - This way, the more "unique" n-grams are given higher weightage in the calculated distance

    :param number_of_truth_titles: Total number of titles in the "truth" data base
    :param max_intersection_possible: The sum of IDFs (Inverse document frequency),
        - for the n-grams of the tile, against which the Jaccard distances are being calculated
    :param non_zero_columns_for_the_row: The row value for the data structure returned by
        - <MatchMaker>._get_matrix_non_zero_columns()
    :param matrix_truth_non_zero_columns_and_values: The data structure returned by
        - <MatchMaker>._get_matrix_truth_non_zero_columns_and_values()
    :param sums_matrix_truth: np.sum(<MatchMaker>.matrix_truth, axis=0)
    :return: The (modified) Jaccard distances
    """
    scores = np.zeros((number_of_truth_titles,), dtype=s.ENCODING_FLOAT_TYPE)
    for non_zero_column in non_zero_columns_for_the_row:
        columns, values = matrix_truth_non_zero_columns_and_values[non_zero_column]
        scores[columns] += values

    return scores / (sums_matrix_truth + (max_intersection_possible - scores))


@numba.njit(parallel=False)
def fast_arg_top_k(array, k):
    """
    Gets the indexes of the top k values in an array.
    * NOTE: The returned indexes are not sorted based on the top values
    * 50x faster than np.argsort
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
    """
    The class responsible for getting the closest (based on Jaccard distance) titles, given a collection of titles.

    :param data: (dataframe) The collection of titles for which to find the closest titles
    :param truth_data: (dataframe) The "truth" database
    :param top_n: (int) Top n values to fetch

    * Main public method: get_closest_matches(...)
    """
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
        """
        Returns a (numba.typed for compatibility with njit mode) list with the non-zero indexes of each row in,
            - self.matrix
        """
        matrix_non_zero_columns = numba.typed.List()
        for row in range(self.matrix.shape[0]):
            matrix_non_zero_columns.append(self.matrix[row].nonzero()[1])

        return matrix_non_zero_columns

    def _get_matrix_truth_non_zero_columns_and_values(self):
        """
        Returns a (numba.typed for compatibility with njit mode) list with,
            - the non-zero indexes of each row, along with the corresponding values, for self.matrix_truth
        """
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
            matrix = lil_matrix(matrix.T, dtype=s.ENCODING_FLOAT_TYPE)

        LOGGER.info(f'[{self.__class__.__name__}] Constructed sparse matrix - {matrix.shape}!')

        return matrix

    def _get_idf_given_index(self, index):
        return self.idf_s_mapping.get(self.n_grams_decoding[index], self.max_idf_value)

    def _get_top_n_matches(self, modified_jaccard):
        """
        For the calculated Jaccard distances, gets the nearest (self.top_n) title_id's from self.truth_data
        """
        top_matches = fast_arg_top_k(modified_jaccard, self.top_n)
        if top_matches.shape[0] != self.top_n:
            raise Exception('top_matches.shape[0] != self.top_n')
        return self.truth_data.loc[top_matches, c.COLUMN_TITLE_ID].tolist()

    def get_closest_matches(self, row_number):
        """
        Given the "row_number" of self.data, gets the closest (self.top_n) titles in self.truth_data
        """
        non_zero_columns_for_the_row = self.matrix_non_zero_columns[row_number]
        max_intersection_possible = sum([self._get_idf_given_index(r) for r in non_zero_columns_for_the_row])

        modified_jaccard = fast_jaccard(
            self.number_of_truth_titles, max_intersection_possible, non_zero_columns_for_the_row,
            self.matrix_truth_non_zero_columns_and_values, self.sums_matrix_truth
        )
        return self._get_top_n_matches(modified_jaccard)
