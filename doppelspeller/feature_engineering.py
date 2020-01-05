import logging
import math

import numba
import numpy as np

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.feature_engineering_prepare import get_closest_matches_per_training_row, generate_misspelled_name
from doppelspeller.common import get_ground_truth, get_train_data, get_test_data, get_words_counter, get_single_data

LOGGER = logging.getLogger(__name__)

DATA_TYPE_MAPPING = {
    c.DATA_TYPE_TRAIN: get_train_data,
    c.DATA_TYPE_TEST: get_test_data,
    c.DATA_TYPE_SINGLE: get_single_data,
}


@numba.njit(numba.uint8(numba.uint8[:], numba.uint8[:]), fastmath=True)
def fast_levenshtein(seq1, seq2):
    length_x = seq1.shape[0]
    length_y = seq2.shape[0]
    total_length = length_x + length_y

    if length_x > length_y:
        length_x, length_y = length_y, length_x
        seq1, seq2 = seq2, seq1

    size_x = length_x + 1
    size_y = length_y + 1

    matrix = np.zeros((size_x, size_y), dtype=s.NUMBER_OF_CHARACTERS_DATA_TYPE)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 2,
                    matrix[x, y - 1] + 1
                )

    return ((total_length - matrix[length_x, length_y]) / total_length) * 100


# 6 Basic + (4 * s.NUMBER_OF_WORDS_FEATURES) "words" related features
FEATURES_COUNT = 6 + (4 * s.NUMBER_OF_WORDS_FEATURES)

signature = [
    (numba.uint8, numba.uint8,
     numba.uint8[:], numba.uint8[:], numba.uint32[:],
     numba.uint8, numba.uint32,
     numba.uint8[:], numba.float32[:])
]
@numba.guvectorize(signature,
                   '(),(),(l),(l),(m),(),(),(n)->(n)', fastmath=True, target='parallel')
def construct_features(title_number_of_characters, truth_number_of_characters,
                       title, title_truth, word_counter,
                       space_code, number_of_truth_titles,
                       dummy, response):

    title = title[:title_number_of_characters]
    title_truth = title_truth[:truth_number_of_characters]

    title_number_of_words = title[title == space_code].shape[0] + 1
    truth_number_of_words = title_truth[title_truth == space_code].shape[0] + 1
    lev_ratio = fast_levenshtein(title, title_truth)

    title_wo_spaces = title[title != space_code]

    title_truth_w_extra_space = np.concatenate(
        (title_truth, np.array([space_code], dtype=s.NUMBER_OF_CHARACTERS_DATA_TYPE)))

    # "Truth" words features
    space_indexes_truth_words = (title_truth_w_extra_space == space_code).nonzero()[0][:s.NUMBER_OF_WORDS_FEATURES]
    reconstructed_title = np.array([space_code], dtype=s.NUMBER_OF_CHARACTERS_DATA_TYPE)
    best_ratios = np.zeros((s.NUMBER_OF_WORDS_FEATURES,), dtype=s.ENCODING_FLOAT_TYPE)
    word_lengths = np.zeros((s.NUMBER_OF_WORDS_FEATURES,), dtype=s.ENCODING_FLOAT_TYPE)
    idf_s = np.zeros((s.NUMBER_OF_WORDS_FEATURES,), dtype=s.ENCODING_FLOAT_TYPE)

    # Assigning nulls
    best_ratios[:] = np.nan
    word_lengths[:] = np.nan
    idf_s[:] = np.nan

    # Truth words loop
    last_index = None
    word_index = -1
    for space_index in space_indexes_truth_words:
        word_index += 1
        if last_index is None:
            truth_word = title_truth[:space_index]
        else:
            truth_word = title_truth[last_index:space_index]
        last_index = space_index + 1

        # Possible words loop
        length_truth_word = truth_word.shape[0]
        best_ratio = 0
        best_match = np.array([space_code], dtype=s.NUMBER_OF_CHARACTERS_DATA_TYPE)
        for possible_index in range(title_wo_spaces.shape[0]):
            possible_word = title_wo_spaces[possible_index:possible_index + length_truth_word]
            if possible_word.shape[0] == 0:
                break

            possible_word_lev_ratio = fast_levenshtein(possible_word, truth_word)
            if possible_word_lev_ratio > best_ratio:
                best_ratio = int(possible_word_lev_ratio)
                best_match = possible_word

        best_ratios[word_index] = best_ratio
        word_lengths[word_index] = truth_word.shape[0]
        idf_s[word_index] = math.log(number_of_truth_titles / word_counter[word_index])
        reconstructed_title = np.concatenate(
            (reconstructed_title, best_match, np.array([space_code], dtype=s.NUMBER_OF_CHARACTERS_DATA_TYPE)))

    # IDF Ranks
    ranks_idf_s = 1 + ((np.nanmax(idf_s) - idf_s) / truth_number_of_words)

    # Removing first and last space
    reconstructed_lev_ratio = fast_levenshtein(reconstructed_title[1: reconstructed_title.shape[0] - 1], title_truth)

    basic_features = np.array([
        title_number_of_characters, truth_number_of_characters,
        title_number_of_words, truth_number_of_words,
        lev_ratio, reconstructed_lev_ratio], dtype=s.ENCODING_FLOAT_TYPE)

    response[:] = np.concatenate((basic_features, best_ratios, word_lengths, idf_s, ranks_idf_s))


class FeatureEngineering:
    def __init__(self, data_type, title=None):
        LOGGER.info(f'[{self.__class__.__name__}] Loading pre-requisite data!')

        data_args = tuple()
        if title:
            data_args = (title,)

        self.data = DATA_TYPE_MAPPING[data_type](*data_args)
        self.truth_data = get_ground_truth()

        self.words_counter = get_words_counter(self.truth_data)
        self.number_of_truth_titles = len(self.truth_data)

        self.allowed_characters = f'{s.R_FILL_CHARACTER} abcdefghijklmnopqrstuvwxyz0123456789'
        self.encoding = {character: index for index, character in enumerate(self.allowed_characters)}
        self.decoding = {value: key for key, value in self.encoding.items()}
        self.space_code = self.encoding[' ']
        if self.encoding[s.R_FILL_CHARACTER] != s.R_FILL_CHARACTER_ENCODING:
            raise Exception('self.encoding[s.R_FILL_CHARACTER] != s.R_FILL_CHARACTER_ENCODING')

    def _generate_dummy_train_data(self):
        LOGGER.info('Generating dummy train data!')

        # Filtering short titles
        generated_training_data = self.truth_data.loc[
            self.truth_data[c.COLUMN_TRANSFORMED_TITLE].str.len() > 9, :].copy(deep=True)

        generated_training_data.loc[:, c.COLUMN_GENERATED_MISSPELLED_TITLE] = \
            generated_training_data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
            lambda x: generate_misspelled_name(x)
        )

        columns_to_include = [c.COLUMN_GENERATED_MISSPELLED_TITLE, c.COLUMN_TRANSFORMED_TITLE]
        generated_training_data = generated_training_data.loc[:, columns_to_include]

        return generated_training_data.reset_index()

    def _prepare_training_input_data(self):
        generated_training_data = self._generate_dummy_train_data()
        training_data_input = get_closest_matches_per_training_row(self.data, self.truth_data)

        training_data_negative = training_data_input.pop(s.TRAIN_NOT_FOUND_VALUE)

        ground_truth_mapping = self.truth_data.set_index(c.COLUMN_TITLE_ID).copy(deep=True)
        ground_truth_mapping = ground_truth_mapping.to_dict()[c.COLUMN_TRANSFORMED_TITLE]

        train_data = self.data.copy(deep=True)
        train_data.loc[:, c.COLUMN_TRAIN_INDEX] = list(train_data.index)
        train_data = train_data.set_index(c.COLUMN_TITLE_ID)
        del train_data[c.COLUMN_TITLE]
        train_data_mapping = train_data.to_dict()[c.COLUMN_TRANSFORMED_TITLE]

        train_data_negatives_mapping = train_data[train_data.index == s.TRAIN_NOT_FOUND_VALUE].copy(deep=True)
        train_data_negatives_mapping = train_data_negatives_mapping.set_index(
            c.COLUMN_TRAIN_INDEX).to_dict()[c.COLUMN_TRANSFORMED_TITLE]

        training_rows_generated = []
        for truth_title, title in zip(generated_training_data[c.COLUMN_TRANSFORMED_TITLE],
                                      generated_training_data[c.COLUMN_GENERATED_MISSPELLED_TITLE]):
            training_rows_generated.append(
                (c.TRAINING_KIND_GENERATED, title, truth_title, 1))

        training_rows_negative = []
        for train_index, titles in training_data_negative.items():
            title = train_data_negatives_mapping[train_index]
            for truth_title_id in titles:
                truth_title = ground_truth_mapping[truth_title_id]
                training_rows_negative.append(
                    (c.TRAINING_KIND_NEGATIVE, title, truth_title, 0))

        training_rows = []
        for title_id, titles in training_data_input.items():
            title = train_data_mapping[title_id]
            for truth_title_id in titles:
                truth_title = ground_truth_mapping[truth_title_id]
                training_rows.append(
                    (c.TRAINING_KIND_POSITIVE, title, truth_title, int(title_id == truth_title_id)))

        return training_rows_negative + training_rows + training_rows_generated

    @staticmethod
    def _get_evaluation_indexes(kind):
        number_of_rows = len(kind)

        evaluation_generated_size = int(number_of_rows * s.EVALUATION_FRACTION_GENERATED_DATA)
        evaluation_negative_size = int(number_of_rows * s.EVALUATION_FRACTION_NEGATIVE_DATA)
        evaluation_positive_size = int(number_of_rows * s.EVALUATION_FRACTION_POSITIVE_DATA)

        candidates_generated_index = (kind == c.TRAINING_KIND_GENERATED).nonzero()[0]
        candidates_negative_index = (kind == c.TRAINING_KIND_NEGATIVE).nonzero()[0]
        candidates_positive_index = (kind == c.TRAINING_KIND_POSITIVE).nonzero()[0]

        evaluation_generated_index = np.random.choice(
            candidates_generated_index, size=evaluation_generated_size, replace=False)
        evaluation_negative_index = np.random.choice(
            candidates_negative_index, size=evaluation_negative_size, replace=False)
        evaluation_positive_index = np.random.choice(
            candidates_positive_index, size=evaluation_positive_size, replace=False)

        return np.array(list(
            set(list(evaluation_generated_index) + list(evaluation_negative_index) + list(evaluation_positive_index))))

    def encode_title(self, title):
        max_allowed_characters = s.MAX_CHARACTERS_ALLOWED_IN_THE_TITLE
        title = title.ljust(max_allowed_characters, s.R_FILL_CHARACTER)[:max_allowed_characters]
        return np.array([self.encoding[x] for x in title], dtype=s.NUMBER_OF_CHARACTERS_DATA_TYPE)

    def encode_word_counter(self, title):
        words = title.split()[:s.NUMBER_OF_WORDS_FEATURES]
        encoded = np.zeros((s.NUMBER_OF_WORDS_FEATURES,), dtype=np.uint32)
        counts = [self.words_counter[x] for x in words]
        encoded[:len(counts)] = counts
        return encoded

    def generate_train_and_evaluation_data_sets(self):
        training_rows_final = self._prepare_training_input_data()

        LOGGER.info('Preparing features!')

        # TODO: Need optimization

        number_of_rows = len(training_rows_final)

        encoding_type = s.NUMBER_OF_CHARACTERS_DATA_TYPE
        float_type = s.ENCODING_FLOAT_TYPE

        title_number_of_characters = np.array([len(x[1]) for x in training_rows_final], dtype=encoding_type)
        truth_number_of_characters = np.array([len(x[2]) for x in training_rows_final], dtype=encoding_type)
        kind = np.array([x[0] for x in training_rows_final], dtype=encoding_type)
        target = np.array([x[3] for x in training_rows_final], dtype=float_type)
        title_encoded = np.array([self.encode_title(x[1]) for x in training_rows_final], dtype=encoding_type)
        title_truth_encoded = np.array([self.encode_title(x[2]) for x in training_rows_final], dtype=encoding_type)
        word_counter_encoded = np.array(
            [self.encode_word_counter(x[2]) for x in training_rows_final], dtype=encoding_type)

        del training_rows_final

        features = np.zeros((number_of_rows, FEATURES_COUNT), dtype=float_type)
        dummy = np.zeros((FEATURES_COUNT,), dtype=encoding_type)

        LOGGER.info(f'Constructing features!')

        construct_features(title_number_of_characters, truth_number_of_characters,
                           title_encoded, title_truth_encoded, word_counter_encoded,
                           self.space_code, self.number_of_truth_titles,
                           dummy, features)

        LOGGER.info(f'Features (shape = {features.shape}) constructed!')

        evaluation_indexes = self._get_evaluation_indexes(kind)
        train_indexes = [i for i in range(number_of_rows) if i not in evaluation_indexes]

        train = features[train_indexes]
        train_target = target[train_indexes]
        evaluation = features[evaluation_indexes]
        evaluation_target = target[evaluation_indexes]

        return (
            train,
            train_target,
            evaluation,
            evaluation_target
        )
