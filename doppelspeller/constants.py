# Columns
COLUMN_WORDS = 'words'
COLUMN_TITLE_ID = 'title_id'
COLUMN_TITLE = 'title'
COLUMN_TRANSFORMED_TITLE = 'transformed_title'
COLUMN_NUMBER_OF_WORDS = 'number_of_words'
COLUMN_N_GRAMS = 'n_grams'
COLUMN_SEQUENCES_MIN_HASH = 'sequences_min_hash'
COLUMN_NUMBER_OF_CHARACTERS = 'number_of_characters'

COLUMN_TRUTH_WORDS = 'truth_words'
COLUMN_TRUTH_TITLE_ID = 'truth_title_id'
COLUMN_TRUTH_TITLE = 'truth_title'
COLUMN_TRUTH_TRANSFORMED_TITLE = 'truth_transformed_title'
COLUMN_TRUTH_NUMBER_OF_WORDS = 'truth_number_of_words'
COLUMN_TRUTH_SEQUENCES = 'truth_sequences'
COLUMN_TRUTH_SEQUENCES_MIN_HASH = 'truth_sequences_min_hash'
COLUMN_TRUTH_NUMBER_OF_CHARACTERS = 'truth_number_of_characters'

COLUMN_TRUTH_TH_WORD_LENGTH = 'truth_{}th_word_length'
COLUMN_TRUTH_TH_WORD_PROBABILITY = 'truth_{}th_word_probability'
COLUMN_TRUTH_TH_WORD_PROBABILITY_RANK = 'truth_{}th_word_probability_rank'
COLUMN_TRUTH_TH_WORD_BEST_MATCH_SCORE = 'truth_{}th_word_best_match_score'

COLUMN_TEST_INDEX = 'test_index'
COLUMN_EXACT = 'exact'
COLUMN_TRAIN_KIND = 'kind'
COLUMN_DISTANCE = 'levenshtein'
COLUMN_TRAIN_INDEX_COLUMN = 'train_index'
COLUMN_GENERATED_MISSPELLED_TITLE = 'generated_misspelled_title'
COLUMN_TARGET = 'target'
COLUMN_RECONSTRUCTED_SCORE = 'reconstructed_score'


TRAINING_KIND_GENERATED = 1
TRAINING_KIND_NEGATIVE = 2
TRAINING_KIND_POSITIVE = 3

DATA_TYPE_TRAIN = 'train'
DATA_TYPE_TEST = 'test'
