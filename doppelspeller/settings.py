import os
import warnings

import doppelspeller.constants as c

PROJECT_DATA_PATH = os.environ.get('PROJECT_DATA_PATH')
if not PROJECT_DATA_PATH:
    PROJECT_DATA_PATH = os.path.abspath('./data/')
    warnings.warn(f'\n\nEnvironment variable PROJECT_DATA_PATH not set! Using {PROJECT_DATA_PATH} as default!\n')
PROJECT_DATA_PATH = os.path.abspath(PROJECT_DATA_PATH)

# Clustering settings
N_GRAMS = 3
TRAIN_DATA_NEAREST_N = 10
SIMILAR_TITLES_FILE = f'{PROJECT_DATA_PATH}/similar_{TRAIN_DATA_NEAREST_N}_titles.dump'
GENERATED_TRAINING_DATA_FILE = f'{PROJECT_DATA_PATH}/generated_training_data.dump'

# Ground Truth input file settings
GROUND_TRUTH_FILE = f'{PROJECT_DATA_PATH}/G.csv'
GROUND_TRUTH_FILE_DELIMITER = '|'
GROUND_TRUTH_FILE_COLUMNS_MAPPING = {
    c.COLUMN_TITLE_ID: 'company_id',
    c.COLUMN_TITLE: 'name'
}

# Train data input file settings
TRAIN_FILE = f'{PROJECT_DATA_PATH}/STrain.csv'
TRAIN_FILE_DELIMITER = '|'
TRAIN_FILE_COLUMNS_MAPPING = {
    c.COLUMN_TITLE_ID: 'company_id',
    c.COLUMN_TITLE: 'name'
}
TRAIN_NOT_FOUND_VALUE = '-1'

# TEST data input file settings
TEST_FILE = f'{PROJECT_DATA_PATH}/STest.csv'
TEST_FILE_DELIMITER = '|'
TEST_FILE_COLUMNS_MAPPING = {
    c.COLUMN_TITLE: 'name'
}

# LSH forest settings
NUMBER_OF_PERMUTATIONS = 128
LSH_FOREST_OUTPUT_FILE = f'{PROJECT_DATA_PATH}/lsh_forest.dump'

# Training settings
MODEL_DUMP_FILE = f'{PROJECT_DATA_PATH}/model.dump'
