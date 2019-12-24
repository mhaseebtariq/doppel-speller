import random
import logging
import math
import sys
import _pickle as pickle
from concurrent.futures import ProcessPoolExecutor

import psutil
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import (get_ground_truth_words_counter, get_train_data, get_min_hash,
                                  get_ground_truth, tf_idf, wait_for_multiprocessing_threads, get_number_of_cpu_workers)


LOGGER = logging.getLogger(__name__)
GROUND_TRUTH, WORDS_COUNTER, NUMBER_OF_TITLES, NUMBER_OF_TITLES = None, None, None, None


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


def populate_pre_requisite_data():
    global GROUND_TRUTH, WORDS_COUNTER, NUMBER_OF_TITLES, NUMBER_OF_TITLES

    GROUND_TRUTH = get_ground_truth()
    WORDS_COUNTER = get_ground_truth_words_counter(GROUND_TRUTH)
    NUMBER_OF_TITLES = len(GROUND_TRUTH)


def prepare_training_input_data():
    populate_pre_requisite_data()

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
    for train_index, row in generated_training_data.iterrows():
        truth_title = row[c.COLUMN_TRANSFORMED_TITLE]
        title = row[c.COLUMN_GENERATED_MISSPELLED_TITLE]
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


def get_evaluation_indexes(features):
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


def construct_features(kind, title, truth_title, target, n=s.NUMBER_OF_WORDS_FEATURES):
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
    tf_idf_s = [tf_idf(x, WORDS_COUNTER, NUMBER_OF_TITLES) for x in truth_words]
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

    return tuple(
        feature_row +
        (word_lengths + extra_nans) +
        (tf_idf_s + extra_nans) +
        (tf_idf_s_ranks + extra_nans) +
        (best_scores + extra_nans) +
        [reconstructed_score, target]
    )


def generate_train_and_evaluation_data_sets():
    training_rows_final = prepare_training_input_data()

    memory_required = sys.getsizeof(np.zeros((1,), dtype=s.FEATURES_TYPES)) * len(training_rows_final)
    memory_required = round(memory_required / 1024 / 1024 / 1024, 2)
    memory_available = round(psutil.virtual_memory().available / 1024 / 1024 / 1024, 2)
    if memory_available < memory_required:
        raise Exception(f'memory_available ({memory_available}GB) < memory_required ({memory_required}GB) '
                        'to build the features matrix! Convert the array using https://www.h5py.org/')

    LOGGER.info('Constructing features!')

    executor = ProcessPoolExecutor(max_workers=get_number_of_cpu_workers())
    number_of_rows = len(training_rows_final)
    threads = [
        executor.submit(construct_features, kind, title, truth_title, target)
        for kind, title, truth_title, target in training_rows_final
    ]
    del training_rows_final
    wait_for_multiprocessing_threads(threads)

    features = np.zeros((number_of_rows,), dtype=s.FEATURES_TYPES)
    features[:] = np.nan

    for index, thread in enumerate(threads):
        features[index] = thread.result()

    evaluation_indexes = get_evaluation_indexes(features)

    evaluation = features[evaluation_indexes]
    train = features[~evaluation_indexes]

    train[c.COLUMN_TRAIN_KIND] = np.nan
    evaluation[c.COLUMN_TRAIN_KIND] = np.nan

    train_target = np.copy(train[c.COLUMN_TARGET])
    evaluation_target = np.copy(evaluation[c.COLUMN_TARGET])

    train[c.COLUMN_TARGET] = np.nan
    evaluation[c.COLUMN_TARGET] = np.nan

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
