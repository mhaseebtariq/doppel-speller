import random
import logging
import math
import _pickle as pickle

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import (get_ground_truth_words_counter, get_train_data, get_min_hash,
                                  get_ground_truth, word_probability)


LOGGER = logging.getLogger(__name__)


def prepare_data_for_features_generation():
    LOGGER.info('Loading LSH forest!')

    with open(s.LSH_FOREST_OUTPUT_FILE, 'rb') as file_object:
        forest = pickle.load(file_object)

    train_data = get_train_data()

    train_data.loc[:, c.COLUMN_SEQUENCES_MIN_HASH] = train_data.loc[:, c.COLUMN_SEQUENCES].apply(
        lambda x: get_min_hash(x, s.NUMBER_OF_PERMUTATIONS))

    train_length = len(train_data)
    similar_titles = {s.TRAIN_NOT_FOUND_VALUE: {}}
    for index, row in train_data.iterrows():
        title_id = row[c.COLUMN_TITLE_ID]
        sequences_min_hash = row[c.COLUMN_SEQUENCES_MIN_HASH]
        matches = forest.query(sequences_min_hash, s.TRAIN_DATA_NEAREST_N)

        if title_id == s.TRAIN_NOT_FOUND_VALUE:
            similar_titles[title_id][index] = matches
            continue
        else:
            similar_titles[title_id] = matches

        if title_id not in matches:
            if len(similar_titles[title_id]) == s.TRAIN_DATA_NEAREST_N:
                similar_titles[title_id].pop()

            similar_titles[title_id].append(title_id)

        if not ((index+1) % 10000):
            LOGGER.info(f'Processed {index+1} of {train_length}...!')

    with open(s.SIMILAR_TITLES_FILE, 'wb') as file_object:
        pickle.dump(similar_titles, file_object, protocol=s.PICKLE_PROTOCOL)

    return s.SIMILAR_TITLES_FILE


KEYBOARD_CARTESIAN = {
    'q': {'x': 0, 'y': 0}, 'w': {'x': 1, 'y': 0}, 'e': {'x': 2, 'y': 0}, 'r': {'x': 3, 'y': 0},
    't': {'x': 4, 'y': 0}, 'y': {'x': 5, 'y': 0}, 'u': {'x': 6, 'y': 0}, 'i': {'x': 7, 'y': 0},
    'o': {'x': 8, 'y': 0}, 'p': {'x': 9, 'y': 0}, 'a': {'x': 0, 'y': 1}, 'z': {'x': 0, 'y': 2},
    's': {'x': 1, 'y': 1}, 'x': {'x': 1, 'y': 2}, 'd': {'x': 2, 'y': 1}, 'c': {'x': 2, 'y': 2},
    'f': {'x': 3, 'y': 1}, 'b': {'x': 4, 'y': 2}, 'm': {'x': 5, 'y': 2}, 'j': {'x': 6, 'y': 1},
    'g': {'x': 4, 'y': 1}, 'h': {'x': 5, 'y': 1}, 'k': {'x': 7, 'y': 1}, 'l': {'x': 8, 'y': 1},
    'v': {'x': 3, 'y': 2}, 'n': {'x': 5, 'y': 2}
}


def euclidean_distance(a, b):
    X = (KEYBOARD_CARTESIAN[a]['x'] - KEYBOARD_CARTESIAN[b]['x']) ** 2
    Y = (KEYBOARD_CARTESIAN[a]['y'] - KEYBOARD_CARTESIAN[b]['y']) ** 2
    return math.sqrt(X + Y)


def get_euclidean_neighbours():
    neighbours = {}
    for i in KEYBOARD_CARTESIAN.keys():
        for j in KEYBOARD_CARTESIAN.keys():
            if i == j:
                continue
            distance = euclidean_distance(i, j)
            if distance <= 1:
                if i in neighbours:
                    neighbours[i].add(j)
                else:
                    neighbours[i] = {j}

                if j in neighbours:
                    neighbours[j].add(i)
                else:
                    neighbours[j] = {i}

    return {k: list(v) for k, v in neighbours.items()}


EUCLIDEAN_NEIGHBOURS = get_euclidean_neighbours()


def remove_letter(x, length):
    index = random.randint(0, length - 1)
    letter = x[index]
    count = 0
    while letter == ' ':
        count += 1
        if count > 10:
            return x
        index = random.randint(0, length - 1)
        letter = x[index]
    return x[:index] + x[index + 1:]


def add_letter(x, length):
    index = random.randint(0, length - 1)
    letter = x[index]
    count = 0
    while letter in ' 0123456789':
        count += 1
        if count > 10:
            return x
        index = random.randint(0, length - 1)
        letter = x[index]
    to_add_options = EUCLIDEAN_NEIGHBOURS[letter]
    return x[:index] + random.choice(to_add_options) + x[index:]


def replace_letter(x, length):
    index = random.randint(0, length - 1)
    letter = x[index]
    count = 0
    while letter in ' 0123456789':
        count += 1
        if count > 10:
            return x
        index = random.randint(0, length - 1)
        letter = x[index]

    return x[:index] + random.choice(EUCLIDEAN_NEIGHBOURS[x[index]]) + x[index + 1:]


def add_space(x, length):
    index = random.randint(1, length - 1)
    check = any([x[index] == ' ', x[index - 1: index] in ('', ' '), x[index + 1: index + 2] in ('', ' ')])
    count = 0
    while check:
        count += 1
        if count > 10:
            return x
        index = random.randint(1, length - 1)
        check = any([x[index] == ' ', x[index - 1: index] in ('', ' '), x[index + 1: index + 2] in ('', ' ')])

    return x[:index] + ' ' + x[index:]


def remove_space(x, length):
    list_x = list(x)
    space_indexes = [index for index, item in enumerate(list_x) if item == ' ']
    if not space_indexes:
        return x
    to_remove = random.choice(space_indexes)
    del list_x[to_remove]

    return ''.join(list_x)


def generate_misspelled_name(word):
    new_word = str(word)
    functions = [random.choice([add_letter, remove_letter]), replace_letter,
                 random.choice([add_space, remove_space])]
    functions_selected = random.sample(functions, random.randint(1, 2))
    for func in functions_selected:
        new_word = func(new_word, len(new_word))
    return new_word


def generate_dummy_train_data():
    LOGGER.info('Generating dummy train data!')

    ground_truth = get_ground_truth()
    ground_truth.loc[:, c.COLUMN_NUMBER_OF_CHARACTERS] = ground_truth.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
        lambda x: len(x)
    )

    # Filtering short titles
    generated_training_data = ground_truth[ground_truth[c.COLUMN_NUMBER_OF_CHARACTERS] > 9].copy(deep=True)

    generated_training_data.loc[:, c.COLUMN_GENERATED_MISSPELLED_TITLE] = \
        generated_training_data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
        lambda x: generate_misspelled_name(x)
    )

    columns_to_include = [c.COLUMN_GENERATED_MISSPELLED_TITLE, c.COLUMN_TRANSFORMED_TITLE]
    generated_training_data = generated_training_data.loc[:, columns_to_include]

    generated_training_data.reset_index().to_pickle(s.GENERATED_TRAINING_DATA_FILE, protocol=s.PICKLE_PROTOCOL)

    return s.GENERATED_TRAINING_DATA_FILE


def construct_features(number_truth_words, truth_words, title_to_match, words_counter, number_of_words, n=15):
    title_to_match = title_to_match.replace(' ', '')
    range_title_to_match = range(len(title_to_match))

    extra_nans = [np.nan] * (n - number_truth_words)

    truth_words = truth_words[:n]

    word_lengths = [len(x) for x in truth_words]
    word_probabilities = [word_probability(x, words_counter, number_of_words) for x in truth_words]
    word_probabilities_ranks = list(np.argsort(word_probabilities).argsort() + 1)

    best_scores = []
    constructed_title = []
    for length_truth_word, truth_word in zip(word_lengths, truth_words):
        possible_words = list(set([title_to_match[i:i + length_truth_word] for i in range_title_to_match]))
        ratios = [fuzz.ratio(truth_word, i) for i in possible_words]
        arg_max = np.argmax(ratios)
        best_score = ratios[arg_max]
        best_score_match = possible_words[arg_max]
        constructed_title.append(best_score_match)
        best_scores.append(best_score)

    reconstructed_score = fuzz.ratio(' '.join(constructed_title), ' '.join(truth_words))
    return (
            (word_lengths + extra_nans) +
            (word_probabilities + extra_nans) +
            (word_probabilities_ranks + extra_nans) +
            (best_scores + extra_nans) +
            [reconstructed_score]
    )


def generate_train_and_evaluation_data_sets():
    with open(s.SIMILAR_TITLES_FILE, 'rb') as file_object:
        training_data_input = pickle.load(file_object)

    training_data_negative = training_data_input.pop(s.TRAIN_NOT_FOUND_VALUE)

    ground_truth = get_ground_truth()

    ground_truth_mapping = ground_truth.set_index(c.COLUMN_TITLE_ID).copy(deep=True)
    ground_truth_mapping = ground_truth_mapping.to_dict()[c.COLUMN_TRANSFORMED_TITLE]

    train_data = get_train_data()
    train_data.loc[:, 'index'] = list(train_data.index)
    train_data = train_data.set_index(c.COLUMN_TITLE_ID)
    del train_data[c.COLUMN_TITLE]
    train_data_mapping = train_data.to_dict()[c.COLUMN_TRANSFORMED_TITLE]

    train_data_negatives_mapping = train_data[train_data.index == s.TRAIN_NOT_FOUND_VALUE].copy(deep=True)
    train_data_negatives_mapping = train_data_negatives_mapping.set_index('index').to_dict()[c.COLUMN_TRANSFORMED_TITLE]

    generated_training_data = pd.read_pickle(s.GENERATED_TRAINING_DATA_FILE)

    training_rows_generated = []
    for train_index, row in generated_training_data.iterrows():
        truth_title = row[c.COLUMN_TRANSFORMED_TITLE]
        title = row[c.COLUMN_GENERATED_MISSPELLED_TITLE]
        training_rows_generated.append(
            [1, title, title.split(' '), truth_title, truth_title.split(' '), 1])

    training_rows_negative = []
    for train_index, titles in training_data_negative.items():
        title = train_data_negatives_mapping[train_index]
        for truth_title_id in titles:
            truth_title = ground_truth_mapping[truth_title_id]
            truth_title_words = truth_title.split(' ')
            title_words = title.split(' ')
            training_rows_negative.append(
                [2, title, title_words, truth_title, truth_title_words, 0])

    training_rows = []
    for title_id, titles in training_data_input.items():
        title = train_data_mapping[title_id]
        for truth_title_id in titles:
            truth_title = ground_truth_mapping[truth_title_id]
            truth_title_words = truth_title.split(' ')
            title_words = title.split(' ')
            training_rows.append(
                [3, title, title_words, truth_title, truth_title_words,
                 int(title_id == truth_title_id)])

    training_rows_final = training_rows_negative + training_rows + training_rows_generated

    train = pd.DataFrame(training_rows_final, columns=[
        'kind', 'title', 'title_words', 'truth_title', 'truth_title_words', 'target'])

    train.loc[:, 'number_of_characters'] = train.loc[:, 'title'].apply(
        lambda x: len(x)
    )
    train.loc[:, 'truth_number_of_characters'] = train.loc[:, 'truth_title'].apply(
        lambda x: len(x)
    )
    train.loc[:, 'number_of_words'] = train.loc[:, 'title_words'].apply(
        lambda x: len(x)
    )
    train.loc[:, 'truth_number_of_words'] = train.loc[:, 'truth_title_words'].apply(
        lambda x: len(x)
    )
    train.loc[:, 'distance'] = list(
        map(lambda x: fuzz.ratio(x[0], x[1]), zip(train.loc[:, 'title'], train.loc[:, 'truth_title'])))

    LOGGER.info('Constructing features!')
    words_counter = get_ground_truth_words_counter(ground_truth)
    number_of_words = len(words_counter)
    extra_features = list(map(lambda x: construct_features(x[0], x[1], x[2], words_counter, number_of_words), zip(
        train.loc[:, 'truth_number_of_words'],
        train.loc[:, 'truth_title_words'],
        train.loc[:, 'title'])))

    columns = ['truth_{}th_word_length'.format(x + 1) for x in range(15)]
    columns += ['truth_{}th_word_probability'.format(x + 1) for x in range(15)]
    columns += ['truth_{}th_word_probability_rank'.format(x + 1) for x in range(15)]
    columns += ['truth_{}th_word_best_match_score'.format(x + 1) for x in range(15)]
    columns.append('reconstructed_score')

    extra = pd.DataFrame(extra_features, columns=columns)

    train = train.merge(extra, right_index=True, left_index=True)

    evaluation_generated = train.loc[train['kind'] == 1, :].sample(frac=0.05).copy(deep=True)
    evaluation_negative = train.loc[train['kind'] == 2, :].sample(frac=0.1).copy(deep=True)
    evaluation_positive = train.loc[train['kind'] == 3, :].sample(frac=0.05).copy(deep=True)

    evaluation = pd.concat([evaluation_generated, evaluation_negative, evaluation_positive])

    evaluation_indexes = list(evaluation_generated.index) + list(evaluation_negative.index) + list(
        evaluation_positive.index)

    train = train.loc[~train.index.isin(evaluation_indexes), :].copy(deep=True).reset_index(drop=True)

    evaluation = evaluation.reset_index(drop=True)

    del train['kind']
    del evaluation['kind']

    train_set = train.loc[:, ~train.columns.isin(
        ['title', 'title_words', 'truth_title', 'truth_title_words', 'target'])]
    train_set_target = train.loc[:, 'target']

    evaluation_set = evaluation.loc[:, ~evaluation.columns.isin(
        ['title', 'title_words', 'truth_title', 'truth_title_words', 'target'])]
    evaluation_set_target = evaluation.loc[:, 'target']

    train_set.to_pickle(s.TRAIN_OUTPUT_FILE, protocol=s.PICKLE_PROTOCOL)
    train_set_target.to_pickle(s.TRAIN_TARGET_OUTPUT_FILE, protocol=s.PICKLE_PROTOCOL)
    evaluation_set.to_pickle(s.EVALUATION_OUTPUT_FILE, protocol=s.PICKLE_PROTOCOL)
    evaluation_set_target.to_pickle(s.EVALUATION_TARGET_OUTPUT_FILE, protocol=s.PICKLE_PROTOCOL)

    return (
        train_set,
        train_set_target,
        evaluation_set,
        evaluation_set_target
    )
