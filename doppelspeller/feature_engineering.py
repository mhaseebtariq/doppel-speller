import logging
import sys
import zlib
import _pickle as pickle

import psutil
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.feature_engineering_prepare import generate_misspelled_name
from doppelspeller.common import (get_words_counter, get_train_data, get_ground_truth, idf,
                                  run_in_multi_processing_mode)


LOGGER = logging.getLogger(__name__)

# TODO: multiprocessing module can not pickle self.<attributes>, therefore, defining global variables!
# Try to avoid declaring these variables, maybe use pathos?
GROUND_TRUTH, WORDS_COUNTER, NUMBER_OF_TITLES = None, None, None


class FeatureEngineering:
    @staticmethod
    def populate_required_data():
        global GROUND_TRUTH, WORDS_COUNTER, NUMBER_OF_TITLES

        GROUND_TRUTH = get_ground_truth()
        WORDS_COUNTER = get_words_counter(GROUND_TRUTH)
        NUMBER_OF_TITLES = len(GROUND_TRUTH)

        return GROUND_TRUTH, WORDS_COUNTER, NUMBER_OF_TITLES

    @staticmethod
    def _generate_dummy_train_data():
        LOGGER.info('Generating dummy train data!')

        # Filtering short titles
        generated_training_data = GROUND_TRUTH.loc[
            GROUND_TRUTH[c.COLUMN_TRANSFORMED_TITLE].str.len() > 9, :].copy(deep=True)

        generated_training_data.loc[:, c.COLUMN_GENERATED_MISSPELLED_TITLE] = \
            generated_training_data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
            lambda x: generate_misspelled_name(x)
        )

        columns_to_include = [c.COLUMN_GENERATED_MISSPELLED_TITLE, c.COLUMN_TRANSFORMED_TITLE]
        generated_training_data = generated_training_data.loc[:, columns_to_include]

        generated_training_data.reset_index().to_pickle(s.GENERATED_TRAINING_DATA_FILE, protocol=s.PICKLE_PROTOCOL)
        del generated_training_data

        return s.GENERATED_TRAINING_DATA_FILE

    def _prepare_training_input_data(self):
        _ = self.populate_required_data()
        _ = self._generate_dummy_train_data()

        with open(s.SIMILAR_TITLES_FILE, 'rb') as file_object:
            training_data_input = pickle.load(file_object)

        training_data_negative = training_data_input.pop(s.TRAIN_NOT_FOUND_VALUE)

        ground_truth_mapping = GROUND_TRUTH.set_index(c.COLUMN_TITLE_ID).copy(deep=True)
        ground_truth_mapping = ground_truth_mapping.to_dict()[c.COLUMN_TRANSFORMED_TITLE]

        train_data = get_train_data()
        train_data.loc[:, c.COLUMN_TRAIN_INDEX_COLUMN] = list(train_data.index)
        train_data = train_data.set_index(c.COLUMN_TITLE_ID)
        del train_data[c.COLUMN_TITLE]
        train_data_mapping = train_data.to_dict()[c.COLUMN_TRANSFORMED_TITLE]

        train_data_negatives_mapping = train_data[train_data.index == s.TRAIN_NOT_FOUND_VALUE].copy(deep=True)
        train_data_negatives_mapping = train_data_negatives_mapping.set_index(
            c.COLUMN_TRAIN_INDEX_COLUMN).to_dict()[c.COLUMN_TRANSFORMED_TITLE]

        generated_training_data = pd.read_pickle(s.GENERATED_TRAINING_DATA_FILE)

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
    def _get_evaluation_indexes(features):
        number_of_rows = len(features)

        evaluation_generated_size = int(number_of_rows * s.EVALUATION_FRACTION_GENERATED_DATA)
        evaluation_negative_size = int(number_of_rows * s.EVALUATION_FRACTION_NEGATIVE_DATA)
        evaluation_positive_size = int(number_of_rows * s.EVALUATION_FRACTION_POSITIVE_DATA)

        candidates_generated_index = (features[c.COLUMN_TRAIN_KIND] == c.TRAINING_KIND_GENERATED).nonzero()[0]
        candidates_negative_index = (features[c.COLUMN_TRAIN_KIND] == c.TRAINING_KIND_NEGATIVE).nonzero()[0]
        candidates_positive_index = (features[c.COLUMN_TRAIN_KIND] == c.TRAINING_KIND_POSITIVE).nonzero()[0]

        evaluation_generated_index = np.random.choice(
            candidates_generated_index, size=evaluation_generated_size, replace=False)
        evaluation_negative_index = np.random.choice(
            candidates_negative_index, size=evaluation_negative_size, replace=False)
        evaluation_positive_index = np.random.choice(
            candidates_positive_index, size=evaluation_positive_size, replace=False)

        return np.array(list(
            set(list(evaluation_generated_index) + list(evaluation_negative_index) + list(evaluation_positive_index))))

    @staticmethod
    def construct_features(kind, title, truth_title, target, n=s.NUMBER_OF_WORDS_FEATURES, compress=False):
        """
        Constructs a feature row per "title" and "title_truth" strings
        :param kind: "Kind" of feature row see -
            TRAINING_KIND_GENERATED, TRAINING_KIND_NEGATIVE, TRAINING_KIND_POSITIVE (in constants.py)
        :param title: Title to match
        :param truth_title: Title to match against
        :param target: True or False - whether the "title" and "title_truth" are a match or not
        :param n: NUMBER_OF_WORDS_FEATURES to consider while defining idf related features (defined in settings.py)
        :param compress: Whether to compress the output tuple
        :return: A tuple corresponding to FEATURES_TYPES (settings.py)
        """
        feature_row = [kind]

        title_number_of_characters = len(title)
        truth_number_of_characters = len(truth_title)
        title_words = title.split(' ')
        truth_words = truth_title.split(' ')
        title_number_of_words = len(title_words)
        truth_number_of_words = len(truth_words)
        distance = fuzz.ratio(title, truth_title)

        feature_row += [title_number_of_characters, truth_number_of_characters,
                        title_number_of_words, truth_number_of_words, distance]

        title_to_match = title.replace(' ', '')
        range_title_to_match = range(len(title_to_match))

        extra_nans = [np.nan] * (n - truth_number_of_words)

        truth_words = truth_words[:n]

        word_lengths = [len(x) for x in truth_words]
        tf_idf_s = [idf(x, WORDS_COUNTER, NUMBER_OF_TITLES) for x in truth_words]
        tf_idf_s_ranks = [int(x) for x in np.argsort(tf_idf_s).argsort() + 1]

        best_scores = []
        constructed_title = []
        for length_truth_word, truth_word in zip(word_lengths, truth_words):
            possible_words = list({title_to_match[i:i + length_truth_word] for i in range_title_to_match})
            ratios = [fuzz.ratio(truth_word, i) for i in possible_words]
            arg_max = np.argmax(ratios)
            best_score = ratios[arg_max]
            best_score_match = possible_words[arg_max]
            constructed_title.append(best_score_match)
            best_scores.append(best_score)

        reconstructed_score = fuzz.ratio(' '.join(constructed_title), ' '.join(truth_words))

        result = tuple(
            feature_row +
            (word_lengths + extra_nans) +
            (tf_idf_s + extra_nans) +
            (tf_idf_s_ranks + extra_nans) +
            (best_scores + extra_nans) +
            [reconstructed_score, target]
        )

        if compress:
            return zlib.compress(pickle.dumps(result))

        return result

    def generate_train_and_evaluation_data_sets(self):
        training_rows_final = self._prepare_training_input_data()

        memory_required = sys.getsizeof(np.zeros((1,), dtype=s.FEATURES_TYPES)) * len(training_rows_final)
        memory_required = round(memory_required / 1024 / 1024 / 1024, 2)
        memory_available = round(psutil.virtual_memory().available / 1024 / 1024 / 1024, 2)
        if memory_available < memory_required:
            raise Exception(f'memory_available ({memory_available}GB) < memory_required ({memory_required}GB) '
                            'to build the features matrix! Convert the array using https://www.h5py.org/')

        LOGGER.info('Constructing features!')

        number_of_rows = len(training_rows_final)
        features = np.zeros((number_of_rows,), dtype=s.FEATURES_TYPES)

        all_args_kwargs = [([kind, title, truth_title, target], {'compress': True})
                           for kind, title, truth_title, target in training_rows_final]
        del training_rows_final
        for index, features_row in enumerate(run_in_multi_processing_mode(self.construct_features, all_args_kwargs)):
            features[index] = pickle.loads(zlib.decompress(features_row))
        del all_args_kwargs

        evaluation_indexes = self._get_evaluation_indexes(features)

        evaluation = features[evaluation_indexes]
        train = features[~evaluation_indexes]

        train[c.COLUMN_TRAIN_KIND] = s.DISABLE_TRAIN_KIND_VALUE
        evaluation[c.COLUMN_TRAIN_KIND] = s.DISABLE_TRAIN_KIND_VALUE

        train_target = np.copy(train[c.COLUMN_TARGET])
        evaluation_target = np.copy(evaluation[c.COLUMN_TARGET])

        train[c.COLUMN_TARGET] = s.DISABLE_TARGET_VALUE
        evaluation[c.COLUMN_TARGET] = s.DISABLE_TARGET_VALUE

        with open(s.TRAIN_OUTPUT_FILE, 'wb') as fl:
            pickle.dump(train, fl)
        with open(s.TRAIN_TARGET_OUTPUT_FILE, 'wb') as fl:
            pickle.dump(train_target, fl)
        with open(s.EVALUATION_OUTPUT_FILE, 'wb') as fl:
            pickle.dump(evaluation, fl)
        with open(s.EVALUATION_TARGET_OUTPUT_FILE, 'wb') as fl:
            pickle.dump(evaluation_target, fl)

        return (
            train,
            train_target,
            evaluation,
            evaluation_target
        )
