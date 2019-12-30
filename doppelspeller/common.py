import re
import time
import logging
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import psutil
import unicodedata
import pandas as pd
import numpy as np
from datasketch import MinHash

import doppelspeller.constants as c
import doppelspeller.settings as s


LOGGER = logging.getLogger(__name__)

SUBSTITUTE_REGEX = re.compile(r' +')
KEEP_REGEX = re.compile(r'[a-zA-Z0-9\s]')


def transform_title(title, trim_large_titles):
    # Remove accents
    text = unicodedata.normalize('NFD', title)
    text = text.encode('ascii', 'ignore').decode('utf-8').lower().replace('-', ' ')
    text = ''.join(KEEP_REGEX.findall(text))
    # Extract only alphanumeric characters / convert to lower case
    text = SUBSTITUTE_REGEX.sub(' ', text)
    if trim_large_titles:
        max_allowed = np.iinfo(s.NUMBER_OF_CHARACTERS_DATA_TYPE).max + 1
        if len(text) > max_allowed:
            LOGGER.warning(
                'Titles greater than length 256 are not allowed. Trimming the title!\n'
                'This is because of the data types set in FEATURES_TYPES (settings.py: NUMBER_OF_CHARACTERS_DATA_TYPE).'
            )
            text = text[:256]

    return text.strip()


def read_and_transform_input_csv(input_file, input_file_delimiter, file_columns_mapping, trim_large_titles):
    data_read = pd.read_csv(input_file, delimiter=input_file_delimiter)

    data = pd.DataFrame(index=range(len(data_read)))
    for column, mapped in file_columns_mapping.items():
        data.loc[:, column] = list(data_read.loc[:, mapped].astype(str))

    data.loc[:, c.COLUMN_TRANSFORMED_TITLE] = data.loc[:, c.COLUMN_TITLE].apply(
        lambda x: transform_title(x, trim_large_titles))
    data.loc[:, c.COLUMN_WORDS] = data.loc[:, c.COLUMN_TRANSFORMED_TITLE].str.split(' ')
    data.loc[:, c.COLUMN_NUMBER_OF_WORDS] = data.loc[:, c.COLUMN_WORDS].str.len()
    data.loc[:, c.COLUMN_N_GRAMS] = data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
        lambda x: get_n_grams(x, s.N_GRAMS)
    )
    invalid_rows = data.loc[data[c.COLUMN_TRANSFORMED_TITLE].str.len() < s.N_GRAMS, :]
    if not invalid_rows.empty:
        LOGGER.warning(
            f'Titles less than length {s.N_GRAMS} found, after transforming the title. Removing those rows!\n'
        )
        data = data.loc[~data.index.isin(invalid_rows.index), :].copy(deep=True).reset_index(drop=True)

    return data


def get_ground_truth(trim_large_titles=True):
    LOGGER.info(f'Reading and transforming the ground truth data!')

    required_columns_in_mapping = [c.COLUMN_TITLE, c.COLUMN_TITLE_ID]
    if sorted(s.GROUND_TRUTH_FILE_COLUMNS_MAPPING.keys()) != required_columns_in_mapping:
        raise Exception('GROUND_TRUTH_FILE_COLUMNS_MAPPING in settings.py should contain the following keys:\n'
                        f'{required_columns_in_mapping}')

    ground_truth = read_and_transform_input_csv(
        s.GROUND_TRUTH_FILE, s.GROUND_TRUTH_FILE_DELIMITER, s.GROUND_TRUTH_FILE_COLUMNS_MAPPING, trim_large_titles)

    LOGGER.info(f'Read {ground_truth.shape[0]} rows from the ground truth data input!')

    return ground_truth


def get_train_data(trim_large_titles=True):
    LOGGER.info(f'Reading and transforming the train data!')

    required_columns_in_mapping = [c.COLUMN_TITLE, c.COLUMN_TITLE_ID]
    if sorted(s.TRAIN_FILE_COLUMNS_MAPPING.keys()) != required_columns_in_mapping:
        raise Exception('TRAIN_FILE_COLUMNS_MAPPING in settings.py should contain the following keys:\n'
                        f'{required_columns_in_mapping}')

    train_data = read_and_transform_input_csv(
        s.TRAIN_FILE, s.TRAIN_FILE_DELIMITER, s.TRAIN_FILE_COLUMNS_MAPPING, trim_large_titles)

    LOGGER.info(f'Read {train_data.shape[0]} rows from the train data input!')

    return train_data


def get_test_data(trim_large_titles=True):
    LOGGER.info(f'Reading and transforming the test data!')

    required_columns_in_mapping = [c.COLUMN_TITLE]
    if sorted(s.TEST_FILE_COLUMNS_MAPPING.keys()) != required_columns_in_mapping:
        raise Exception('TEST_FILE_COLUMNS_MAPPING in settings.py should contain the following keys:\n'
                        f'{required_columns_in_mapping}')

    test_data = read_and_transform_input_csv(
        s.TEST_FILE, s.TEST_FILE_DELIMITER, s.TEST_FILE_COLUMNS_MAPPING, trim_large_titles)

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


def idf(word, words_counter, number_of_titles):
    """
    Inverse document frequency
    """
    return math.log(number_of_titles / words_counter[word])


def get_min_hash(title, num_perm):
    min_hash = MinHash(num_perm=num_perm)
    _ = [min_hash.update(str(x).encode('utf8')) for x in title]
    return min_hash


def wait_for_multiprocessing_threads(threads):
    LOGGER.info('Waiting for the multi processing threads to complete!')

    all_threads_count = len(threads)
    done_threads = [x for x in threads if x.done()]
    done_threads_count = len(done_threads)
    while done_threads_count != all_threads_count:
        time.sleep(10)
        LOGGER.info(f'Processed {done_threads_count} out of {all_threads_count}...')

        exception_threads = [x for x in threads if x.exception()]
        if exception_threads:
            for thread in threads:
                thread.cancel()
            raise exception_threads[0].exception()

        done_threads = [x for x in threads if x.done()]
        done_threads_count = len(done_threads)

    LOGGER.info('Multi processing threads completed!')


def get_number_of_cpu_workers():
    count = psutil.cpu_count(logical=False) - 1
    if count == 0:
        LOGGER.warning(
            'Running the multiprocessing code with max_workers=1 because the machine is single core. '
            'This can slow things up!'
        )
    return count or 1


def run_in_multi_processing_mode(func, all_args_kwargs):
    number_of_workers = get_number_of_cpu_workers()

    if number_of_workers == 1:
        LOGGER.warning('Starting single threaded process. Use multi core machine to speed this up!')
        result = [func(*args, **kwargs) for args, kwargs in all_args_kwargs]
    else:
        LOGGER.info('Starting multi processing threads!')
        executor = ProcessPoolExecutor(max_workers=number_of_workers)
        threads = [executor.submit(func, *args, **kwargs) for args, kwargs in all_args_kwargs]

        wait_for_multiprocessing_threads(threads)

        result = [thread.result() for thread in threads]
        del threads

    return result
