import re
import time
import logging
import math
from collections import Counter

import psutil
import unicodedata
import pandas as pd
from datasketch import MinHash

import doppelspeller.constants as c
import doppelspeller.settings as s


LOGGER = logging.getLogger(__name__)

SUBSTITUTE_REGEX = re.compile(r'[\s\-]+')
KEEP_REGEX = re.compile(r'[a-zA-Z0-9\s]')


def get_number_of_cpu_workers():
    count = psutil.cpu_count(logical=False) - 1
    if count == 0:
        LOGGER.warning(
            'Running the multiprocessing code with max_workers=1 because the machine is single core. '
            'This can slow things up!'
        )
    return count or 1


def transform_title(title):
    # Remove accents
    text = unicodedata.normalize('NFD', title)
    text = text.encode('ascii', 'ignore').decode('utf-8').lower()
    # Extract only alphanumeric characters / convert to lower case
    text = SUBSTITUTE_REGEX.sub(' ', text).strip()
    text = ''.join(KEEP_REGEX.findall(text))
    if len(text) > 256:
        LOGGER.warning(
            'Titles greater than length 256 are not allowed. Trimming the title!\n'
            'This is because of the data types set in FEATURES_TYPES (settings.py).'
        )
        text = text[:256]
    return text


def read_and_transform_input_csv(input_file, input_file_delimiter, file_columns_mapping):
    data_read = pd.read_csv(input_file, delimiter=input_file_delimiter)

    data = pd.DataFrame(index=range(len(data_read)))
    for column, mapped in file_columns_mapping.items():
        data.loc[:, column] = list(data_read.loc[:, mapped].astype(str))

    data.loc[:, c.COLUMN_TRANSFORMED_TITLE] = data.loc[:, c.COLUMN_TITLE].apply(
        lambda x: transform_title(x))
    data.loc[:, c.COLUMN_WORDS] = data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(lambda x: x.split(' '))
    data.loc[:, c.COLUMN_NUMBER_OF_WORDS] = data.loc[:, c.COLUMN_WORDS].apply(lambda x: len(x))
    data.loc[:, c.COLUMN_SEQUENCES] = data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
        lambda x: get_sequences(x, s.N_GRAMS)
    )

    return data


def get_ground_truth():
    LOGGER.info(f'Reading and transforming the ground truth data!')

    required_columns_in_mapping = [c.COLUMN_TITLE, c.COLUMN_TITLE_ID]
    if sorted(s.GROUND_TRUTH_FILE_COLUMNS_MAPPING.keys()) != required_columns_in_mapping:
        raise Exception('GROUND_TRUTH_FILE_COLUMNS_MAPPING in settings.py should contain the following keys:\n'
                        f'{required_columns_in_mapping}')

    ground_truth = read_and_transform_input_csv(
        s.GROUND_TRUTH_FILE, s.GROUND_TRUTH_FILE_DELIMITER, s.GROUND_TRUTH_FILE_COLUMNS_MAPPING)

    LOGGER.info(f'Read {ground_truth.shape[0]} rows from the ground truth data input!')

    return ground_truth


def get_train_data():
    LOGGER.info(f'Reading and transforming the train data!')

    required_columns_in_mapping = [c.COLUMN_TITLE, c.COLUMN_TITLE_ID]
    if sorted(s.TRAIN_FILE_COLUMNS_MAPPING.keys()) != required_columns_in_mapping:
        raise Exception('TRAIN_FILE_COLUMNS_MAPPING in settings.py should contain the following keys:\n'
                        f'{required_columns_in_mapping}')

    train_data = read_and_transform_input_csv(
        s.TRAIN_FILE, s.TRAIN_FILE_DELIMITER, s.TRAIN_FILE_COLUMNS_MAPPING)

    LOGGER.info(f'Read {train_data.shape[0]} rows from the train data input!')

    return train_data


def get_test_data():
    LOGGER.info(f'Reading and transforming the test data!')

    required_columns_in_mapping = [c.COLUMN_TITLE]
    if sorted(s.TEST_FILE_COLUMNS_MAPPING.keys()) != required_columns_in_mapping:
        raise Exception('TEST_FILE_COLUMNS_MAPPING in settings.py should contain the following keys:\n'
                        f'{required_columns_in_mapping}')

    test_data = read_and_transform_input_csv(
        s.TEST_FILE, s.TEST_FILE_DELIMITER, s.TEST_FILE_COLUMNS_MAPPING)

    LOGGER.info(f'Read {test_data.shape[0]} rows from the test data input!')

    return test_data


def get_ground_truth_words_counter(ground_truth):
    words = [x for y in ground_truth.loc[:, c.COLUMN_WORDS] for x in y]
    return Counter(words)


def get_sequences(words, n_grams):
    return set([words[i:i+n_grams] for i in range(len(words)) if len(words[i:i+n_grams]) == n_grams])


def tf_idf(word, words_counter, number_of_titles):
    tf = words_counter[word] / number_of_titles
    idf = math.log(number_of_titles / words_counter[word])
    return tf * idf


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

        exception_threads = [x for x in done_threads if x.exception()]
        if exception_threads:
            for thread in threads:
                thread.cancel()
            raise exception_threads[0].exception()

        done_threads = [x for x in threads if x.done()]
        done_threads_count = len(done_threads)

    LOGGER.info('Multi processing threads completed!')
