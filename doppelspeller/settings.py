import os
import warnings

import numpy as np

import doppelspeller.constants as c

PROJECT_DATA_PATH = os.environ.get('PROJECT_DATA_PATH')
if not PROJECT_DATA_PATH:
    PROJECT_DATA_PATH = os.path.abspath('./data/')
    warnings.warn(f'\n\nEnvironment variable PROJECT_DATA_PATH not set! Using {PROJECT_DATA_PATH} as default!\n')
PROJECT_DATA_PATH = os.path.abspath(PROJECT_DATA_PATH)

DISABLE_NUMBA = False

# Pickling settings
PICKLE_PROTOCOL = 3

# Clustering settings
N_GRAMS = 3

# Ground Truth input file settings
GROUND_TRUTH_FILE = f'{PROJECT_DATA_PATH}/example_truth.csv'
GROUND_TRUTH_FILE_DELIMITER = '|'
GROUND_TRUTH_FILE_COLUMNS_MAPPING = {
    c.COLUMN_TITLE_ID: 'company_id',
    c.COLUMN_TITLE: 'name'
}

# Train data input file settings
TRAIN_FILE = f'{PROJECT_DATA_PATH}/example_train.csv'
TRAIN_FILE_DELIMITER = '|'
TRAIN_FILE_COLUMNS_MAPPING = {
    c.COLUMN_TITLE_ID: 'company_id',
    c.COLUMN_TITLE: 'name'
}
TRAIN_NOT_FOUND_VALUE = '-1'

# TEST data input file settings
TEST_FILE = f'{PROJECT_DATA_PATH}/example_test.csv'
TEST_WITH_ACTUAL_FILE = f'{PROJECT_DATA_PATH}/example_test_with_actuals.csv'
TEST_FILE_DELIMITER = '|'
TEST_FILE_COLUMNS_MAPPING = {
    c.COLUMN_TITLE: 'name'
}

# Training settings
MODEL_DUMP_FILE = f'{PROJECT_DATA_PATH}/model.dump'
EVALUATION_FRACTION_GENERATED_DATA = 0.05
EVALUATION_FRACTION_NEGATIVE_DATA = 0.1
EVALUATION_FRACTION_POSITIVE_DATA = 0.05


# Features files
TRAIN_OUTPUT_FILE = f'{PROJECT_DATA_PATH}/train_data.dump'
TRAIN_TARGET_OUTPUT_FILE = f'{PROJECT_DATA_PATH}/train_target_data.dump'
EVALUATION_OUTPUT_FILE = f'{PROJECT_DATA_PATH}/evaluation_data.dump'
EVALUATION_TARGET_OUTPUT_FILE = f'{PROJECT_DATA_PATH}/evaluation_target_data.dump'

# SQLITE settings
SQLITE_DB = f'{PROJECT_DATA_PATH}/data.db'
SQLITE_NEIGHBOURS_TABLE = 'neighbours'
SQLITE_PREDICTIONS_TABLE = 'predictions'
SQLITE_FEATURES_INPUT_TABLE = 'features_input'

TOP_N_RESULTS_TO_FIND_FOR_TRAINING = 10
TOP_N_RESULTS_TO_FIND_FOR_PREDICTING = 1000
NEAREST_TITLES_TRAIN_FILE = f'{PROJECT_DATA_PATH}/train_nearest_{TOP_N_RESULTS_TO_FIND_FOR_TRAINING}_titles.dump'
NEAREST_TITLES_TEST_FILE = f'{PROJECT_DATA_PATH}/test_nearest_{TOP_N_RESULTS_TO_FIND_FOR_PREDICTING}_titles.dump'
SIMILAR_TITLES_FILE = f'{PROJECT_DATA_PATH}/similar_{TOP_N_RESULTS_TO_FIND_FOR_TRAINING}_titles.dump'

# Predictions settings
FINAL_OUTPUT_FILE = f'{PROJECT_DATA_PATH}/final_output.csv'

# Features settings
NUMBER_OF_WORDS_FEATURES = 15
NUMBER_OF_CHARACTERS_DATA_TYPE = np.dtype('u1')
FEATURES_TYPES = [
    (c.COLUMN_TRAIN_KIND, np.dtype('u1')),  # NOT A FEATURE - only used for filtering
    (c.COLUMN_NUMBER_OF_CHARACTERS, NUMBER_OF_CHARACTERS_DATA_TYPE),
    (c.COLUMN_TRUTH_NUMBER_OF_CHARACTERS, NUMBER_OF_CHARACTERS_DATA_TYPE),
    (c.COLUMN_NUMBER_OF_WORDS, np.dtype('u1')),
    (c.COLUMN_TRUTH_NUMBER_OF_WORDS, np.dtype('u1')),
    (c.COLUMN_DISTANCE, np.dtype('u1')),
]
FEATURES_TYPES += [(c.COLUMN_TRUTH_TH_WORD_LENGTH.format(x + 1), np.float16)
                   for x in range(NUMBER_OF_WORDS_FEATURES)]
FEATURES_TYPES += [(c.COLUMN_TRUTH_TH_WORD_PROBABILITY.format(x + 1), np.float16)
                   for x in range(NUMBER_OF_WORDS_FEATURES)]
FEATURES_TYPES += [(c.COLUMN_TRUTH_TH_WORD_PROBABILITY_RANK.format(x + 1), np.float16)
                   for x in range(NUMBER_OF_WORDS_FEATURES)]
FEATURES_TYPES += [(c.COLUMN_TRUTH_TH_WORD_BEST_MATCH_SCORE.format(x + 1), np.float16)
                   for x in range(NUMBER_OF_WORDS_FEATURES)]
FEATURES_TYPES += [(c.COLUMN_RECONSTRUCTED_SCORE, np.dtype('u1')), (c.COLUMN_TARGET, np.bool)]

DISABLE_TRAIN_KIND_VALUE = 0
DISABLE_TARGET_VALUE = np.nan

# Model settings
PREDICTION_PROBABILITY_THRESHOLD = 0.5
FALSE_POSITIVE_PENALTY_FACTOR = 5
