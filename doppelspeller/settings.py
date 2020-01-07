import os
import warnings

import numpy as np

import doppelspeller.constants as c

PROJECT_DATA_PATH = os.environ.get('PROJECT_DATA_PATH')
if not PROJECT_DATA_PATH:
    PROJECT_DATA_PATH = os.path.abspath('./data/')
    warnings.warn(f'\n\nEnvironment variable PROJECT_DATA_PATH not set! Using {PROJECT_DATA_PATH} as default!\n')
PROJECT_DATA_PATH = os.path.abspath(PROJECT_DATA_PATH)

# Jaccard distance settings
N_GRAMS = 3

# Ground Truth input file settings
GROUND_TRUTH_FILE = f'{PROJECT_DATA_PATH}/example_truth.csv'
GROUND_TRUTH_FILE_DELIMITER = '|'
GROUND_TRUTH_FILE_COLUMNS_MAPPING = [
    (c.COLUMN_TITLE_ID, ('company_id', int)),
    (c.COLUMN_TITLE, ('name', str)),
]

# Train data input file settings
TRAIN_FILE = f'{PROJECT_DATA_PATH}/example_train.csv'
TRAIN_FILE_DELIMITER = '|'
TRAIN_FILE_COLUMNS_MAPPING = [
    (c.COLUMN_TRAIN_INDEX, ('train_index', int)),
    (c.COLUMN_TITLE, ('name', str)),
    (c.COLUMN_TITLE_ID, ('company_id', int)),
]
TRAIN_NOT_FOUND_VALUE = -1

# TEST data input file settings
TEST_FILE = f'{PROJECT_DATA_PATH}/example_test.csv'
TEST_WITH_ACTUALS_FILE = f'{PROJECT_DATA_PATH}/example_test_with_actuals.csv'
TEST_WITH_ACTUALS_TITLE_ID = 'company_id'
TEST_FILE_DELIMITER = '|'
TEST_FILE_COLUMNS_MAPPING = [
    (c.COLUMN_TEST_INDEX, ('test_index', int)),
    (c.COLUMN_TITLE, ('name', str)),
]

# Training settings
MODEL_DUMP_FILE = f'{PROJECT_DATA_PATH}/model.pickle'
EVALUATION_FRACTION_GENERATED_DATA = 0.05
EVALUATION_FRACTION_NEGATIVE_DATA = 0.1
EVALUATION_FRACTION_POSITIVE_DATA = 0.05

# SQLITE settings
SQLITE_DB = f'{PROJECT_DATA_PATH}/data.db'
SQLITE_PREDICTIONS_TABLE = 'predictions'

TOP_N_RESULTS_TO_FIND_FOR_TRAINING = 10
TOP_N_RESULTS_TO_FIND_FOR_PREDICTING = 100

if TOP_N_RESULTS_TO_FIND_FOR_TRAINING > TOP_N_RESULTS_TO_FIND_FOR_PREDICTING:
    raise Exception('TOP_N_RESULTS_TO_FIND_FOR_TRAINING can not be greater than TOP_N_RESULTS_TO_FIND_FOR_PREDICTING!')

# Predictions settings
FINAL_OUTPUT_FILE = f'{PROJECT_DATA_PATH}/final_output.csv'

# Features settings
NUMBER_OF_WORDS_FEATURES = 15
WORDS_COUNT_DATA_TYPE = np.uint32
NUMBER_OF_CHARACTERS_DATA_TYPE = np.uint8
MAX_CHARACTERS_ALLOWED_IN_THE_TITLE = np.iinfo(NUMBER_OF_CHARACTERS_DATA_TYPE).max + 1
R_FILL_CHARACTER = '-'
R_FILL_CHARACTER_ENCODING = 0
ENCODING_FLOAT_TYPE = np.float32
ENCODING_FLOAT_BUFFER = np.finfo(ENCODING_FLOAT_TYPE).resolution

# Model settings
PREDICTION_PROBABILITY_THRESHOLD = 0.5
FALSE_POSITIVE_PENALTY_FACTOR = 5
