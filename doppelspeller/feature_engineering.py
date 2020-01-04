import logging
from collections import namedtuple

import numpy as np
from Levenshtein import ratio

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.feature_engineering_prepare import get_closest_matches_per_training_row, generate_misspelled_name
from doppelspeller.common import get_words_counter, load_processed_train_data, load_processed_test_data, idf_word


LOGGER = logging.getLogger(__name__)

DataMapper = namedtuple('DataMapper', ['loader', 'nearest_matches_key'])
DATA_TYPE_MAPPING = {
    c.DATA_TYPE_TRAIN: DataMapper(loader=load_processed_train_data, nearest_matches_key=c.DATA_TYPE_NEAREST_TRAIN),
    c.DATA_TYPE_TEST: DataMapper(loader=load_processed_test_data, nearest_matches_key=c.DATA_TYPE_NEAREST_TEST),
}


def levenshtein_ratio(text, text_to_match):
    return int(round(ratio(text, text_to_match) * 100))


def levenshtein_token_sort_ratio(text, text_to_match):
    text, text_to_match = ' '.join(sorted(text.split())), ' '.join(sorted(text_to_match.split()))
    return levenshtein_ratio(text, text_to_match)


class FeatureEngineering:
    def __init__(self, data_type):
        self.data_mapper = DATA_TYPE_MAPPING[data_type]
        self.processed_data = self.data_mapper.loader()

        self.data = self.processed_data[data_type]
        self.truth_data = self.processed_data[c.DATA_TYPE_TRUTH]
        self.nearest_matches = self.processed_data[self.data_mapper.nearest_matches_key]

        self.words_counter = get_words_counter(self.truth_data)
        self.number_of_titles = len(self.truth_data)

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
        training_data_input = get_closest_matches_per_training_row()

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

    def construct_features(self, kind, title, truth_title, target, n=s.NUMBER_OF_WORDS_FEATURES):
        """
        TODO: Optimize this method!
        Constructs a feature row per "title" and "title_truth" strings
        :param kind: "Kind" of feature row see -
            TRAINING_KIND_GENERATED, TRAINING_KIND_NEGATIVE, TRAINING_KIND_POSITIVE (in constants.py)
        :param title: Title to match
        :param truth_title: Title to match against
        :param target: True or False - whether the "title" and "title_truth" are a match or not
        :param n: NUMBER_OF_WORDS_FEATURES to consider while defining idf_word related features (defined in settings.py)
        :return: A tuple corresponding to FEATURES_TYPES (settings.py)
        """
        feature_row = [kind]

        title_number_of_characters = len(title)
        truth_number_of_characters = len(truth_title)
        title_words = title.split()
        truth_words = truth_title.split()
        title_number_of_words = len(title_words)
        truth_number_of_words = len(truth_words)
        distance = levenshtein_ratio(title, truth_title)

        feature_row += [title_number_of_characters, truth_number_of_characters,
                        title_number_of_words, truth_number_of_words, distance]

        title_to_match = title.replace(' ', '')
        range_title_to_match = range(len(title_to_match))

        extra_nans = [np.nan] * (n - truth_number_of_words)

        truth_words = truth_words[:n]

        word_lengths = [len(x) for x in truth_words]
        idf_s = [idf_word(x, self.words_counter, self.number_of_titles) for x in truth_words]
        idf_s_ranks = [int(x) for x in np.argsort(idf_s).argsort() + 1]

        best_scores = []
        constructed_title = []
        for length_truth_word, truth_word in zip(word_lengths, truth_words):
            possible_words = list({title_to_match[i:i + length_truth_word] for i in range_title_to_match})
            ratios = [levenshtein_ratio(truth_word, i) for i in possible_words]
            arg_max = np.argmax(ratios)
            best_score = ratios[arg_max]
            best_score_match = possible_words[arg_max]
            constructed_title.append(best_score_match)
            best_scores.append(best_score)

        reconstructed_score = levenshtein_ratio(' '.join(constructed_title), ' '.join(truth_words))

        result = tuple(
            feature_row +
            (word_lengths + extra_nans) +
            (idf_s + extra_nans) +
            (idf_s_ranks + extra_nans) +
            (best_scores + extra_nans) +
            [reconstructed_score, target]
        )

        return result

    def generate_train_and_evaluation_data_sets(self):
        training_rows_final = self._prepare_training_input_data()

        LOGGER.info('Constructing features!')

        number_of_rows = len(training_rows_final)
        features = np.zeros((number_of_rows,), dtype=s.FEATURES_TYPES)

        for index, (kind, title, truth_title, target) in enumerate(training_rows_final):
            if not((index+1) % 10000):
                print(f'Processed {index+1} of {number_of_rows}!')
            features[index] = self.construct_features(kind, title, truth_title, target)

        evaluation_indexes = self._get_evaluation_indexes(features)

        evaluation = features[evaluation_indexes]
        train = features[~evaluation_indexes]

        train[c.COLUMN_TRAIN_KIND] = s.DISABLE_TRAIN_KIND_VALUE
        evaluation[c.COLUMN_TRAIN_KIND] = s.DISABLE_TRAIN_KIND_VALUE

        train_target = np.copy(train[c.COLUMN_TARGET])
        evaluation_target = np.copy(evaluation[c.COLUMN_TARGET])

        train[c.COLUMN_TARGET] = s.DISABLE_TARGET_VALUE
        evaluation[c.COLUMN_TARGET] = s.DISABLE_TARGET_VALUE

        return (
            train,
            train_target,
            evaluation,
            evaluation_target
        )
