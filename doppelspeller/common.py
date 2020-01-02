import re
import logging
import math
import _pickle as pickle
from collections import Counter

import unicodedata
import pandas as pd
from datasketch import MinHash

import doppelspeller.constants as c
import doppelspeller.settings as s


LOGGER = logging.getLogger(__name__)

SUBSTITUTE_REGEX = re.compile(r' +')
KEEP_REGEX = re.compile(r'[a-zA-Z0-9\s]')


def transform_title(title):
    # Remove accents
    text = unicodedata.normalize('NFD', title)
    text = text.encode('ascii', 'ignore').decode('utf-8').lower().replace('-', ' ')
    text = ''.join(KEEP_REGEX.findall(text))
    # Extract only alphanumeric characters / convert to lower case
    text = SUBSTITUTE_REGEX.sub(' ', text).strip()
    number_of_characters = len(text)
    text = text[: s.MAX_CHARACTERS_ALLOWED_IN_THE_TITLE].strip()

    if number_of_characters < s.N_GRAMS:
        LOGGER.warning(
            f"Titles less than length {s.N_GRAMS} found, after transforming the title. Pre-pending 0's!\n"
        )
        return text.rjust(s.N_GRAMS, '0')

    elif number_of_characters > s.MAX_CHARACTERS_ALLOWED_IN_THE_TITLE:
        LOGGER.warning(
            'Titles greater than length 256 are not allowed. Trimming the title!\n'
            'This is because of the data types set in FEATURES_TYPES (settings.py: NUMBER_OF_CHARACTERS_DATA_TYPE).'
        )

    return text


def read_and_transform_input_csv(input_file, input_file_delimiter, file_columns_mapping):
    data_read = pd.read_csv(input_file, delimiter=input_file_delimiter)

    data = pd.DataFrame(index=range(len(data_read)))
    for column, (column_source_file, column_type) in file_columns_mapping:
        data.loc[:, column] = list(data_read.loc[:, column_source_file].astype(column_type))

    data.loc[:, c.COLUMN_TRANSFORMED_TITLE] = data.loc[:, c.COLUMN_TITLE].apply(
        lambda x: transform_title(x))
    data.loc[:, c.COLUMN_WORDS] = data.loc[:, c.COLUMN_TRANSFORMED_TITLE].str.split(' ')
    data.loc[:, c.COLUMN_NUMBER_OF_WORDS] = data.loc[:, c.COLUMN_WORDS].str.len()
    data.loc[:, c.COLUMN_N_GRAMS] = data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
        lambda x: get_n_grams(x, s.N_GRAMS)
    )

    return data


def get_ground_truth():
    LOGGER.info(f'Reading and transforming the ground truth data!')

    required_columns_in_mapping = [c.COLUMN_TITLE_ID, c.COLUMN_TITLE]
    if [x[0] for x in s.GROUND_TRUTH_FILE_COLUMNS_MAPPING] != required_columns_in_mapping:
        raise Exception('GROUND_TRUTH_FILE_COLUMNS_MAPPING in settings.py should contain the following keys:\n'
                        f'{required_columns_in_mapping}')

    ground_truth = read_and_transform_input_csv(
        s.GROUND_TRUTH_FILE, s.GROUND_TRUTH_FILE_DELIMITER, s.GROUND_TRUTH_FILE_COLUMNS_MAPPING)

    LOGGER.info(f'Read {ground_truth.shape[0]} rows from the ground truth data input!')

    return ground_truth


def get_train_data():
    LOGGER.info(f'Reading and transforming the train data!')

    required_columns_in_mapping = [c.COLUMN_TRAIN_INDEX, c.COLUMN_TITLE, c.COLUMN_TITLE_ID]
    if [x[0] for x in s.TRAIN_FILE_COLUMNS_MAPPING] != required_columns_in_mapping:
        raise Exception('TRAIN_FILE_COLUMNS_MAPPING in settings.py should contain the following keys:\n'
                        f'{required_columns_in_mapping}')

    train_data = read_and_transform_input_csv(
        s.TRAIN_FILE, s.TRAIN_FILE_DELIMITER, s.TRAIN_FILE_COLUMNS_MAPPING)

    LOGGER.info(f'Read {train_data.shape[0]} rows from the train data input!')

    return train_data


def get_test_data():
    LOGGER.info(f'Reading and transforming the test data!')

    required_columns_in_mapping = [c.COLUMN_TEST_INDEX, c.COLUMN_TITLE]
    if [x[0] for x in s.TEST_FILE_COLUMNS_MAPPING] != required_columns_in_mapping:
        raise Exception('TEST_FILE_COLUMNS_MAPPING in settings.py should contain the following keys:\n'
                        f'{required_columns_in_mapping}')

    test_data = read_and_transform_input_csv(
        s.TEST_FILE, s.TEST_FILE_DELIMITER, s.TEST_FILE_COLUMNS_MAPPING)

    LOGGER.info(f'Read {test_data.shape[0]} rows from the test data input!')

    return test_data


def get_words_counter(data):
    words = [x for y in data.loc[:, c.COLUMN_WORDS] for x in set(y)]
    return Counter(words)


def get_n_grams_counter(data):
    words = [x for y in data.loc[:, c.COLUMN_N_GRAMS] for x in set(y)]
    return Counter(words)


def get_n_grams(title, n_grams):
    return set([title[i:i+n_grams] for i in range(len(title)) if len(title[i:i+n_grams]) == n_grams])


def idf_word(word, words_counter, number_of_titles):
    """
    Inverse document frequency
    """
    return math.log(number_of_titles / words_counter[word])


def get_min_hash(title, num_perm):
    min_hash = MinHash(num_perm=num_perm)
    _ = [min_hash.update(str(x).encode('utf8')) for x in title]
    return min_hash


def load_processed_train_data():
    try:
        with open(s.PRE_REQUISITE_TRAIN_DATA_FILE, 'rb') as fl:
            return pickle.load(fl)
    except FileNotFoundError:
        LOGGER.warning(f'File ({s.PRE_REQUISITE_TRAIN_DATA_FILE}) not found. Please run the "pre-process-data" cli!')


def load_processed_test_data():
    try:
        with open(s.PRE_REQUISITE_TEST_DATA_FILE, 'rb') as fl:
            return pickle.load(fl)
    except FileNotFoundError:
        LOGGER.warning(f'File ({s.PRE_REQUISITE_TEST_DATA_FILE}) not found. Please run the "pre-process-data" cli!')
