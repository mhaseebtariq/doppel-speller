import random
import logging
import math
import _pickle as pickle

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import get_train_data, get_min_hash


LOGGER = logging.getLogger(__name__)

KEYBOARD_CARTESIAN = {
    'q': {'x': 0, 'y': 0}, 'w': {'x': 1, 'y': 0}, 'e': {'x': 2, 'y': 0}, 'r': {'x': 3, 'y': 0},
    't': {'x': 4, 'y': 0}, 'y': {'x': 5, 'y': 0}, 'u': {'x': 6, 'y': 0}, 'i': {'x': 7, 'y': 0},
    'o': {'x': 8, 'y': 0}, 'p': {'x': 9, 'y': 0}, 'a': {'x': 0, 'y': 1}, 'z': {'x': 0, 'y': 2},
    's': {'x': 1, 'y': 1}, 'x': {'x': 1, 'y': 2}, 'd': {'x': 2, 'y': 1}, 'c': {'x': 2, 'y': 2},
    'f': {'x': 3, 'y': 1}, 'b': {'x': 4, 'y': 2}, 'm': {'x': 5, 'y': 2}, 'j': {'x': 6, 'y': 1},
    'g': {'x': 4, 'y': 1}, 'h': {'x': 5, 'y': 1}, 'k': {'x': 7, 'y': 1}, 'l': {'x': 8, 'y': 1},
    'v': {'x': 3, 'y': 2}, 'n': {'x': 5, 'y': 2}
}


def prepare_data_for_features_generation():
    LOGGER.info('Loading LSH forest!')

    with open(s.LSH_FOREST_OUTPUT_FILE, 'rb') as file_object:
        forest = pickle.load(file_object)

    train_data = get_train_data()

    train_data.loc[:, c.COLUMN_SEQUENCES_MIN_HASH] = train_data.loc[:, c.COLUMN_SEQUENCES].apply(
        lambda x: get_min_hash(x, s.NUMBER_OF_PERMUTATIONS))

    train_length = len(train_data)
    similar_titles = {s.TRAIN_NOT_FOUND_VALUE: {}}
    for index, (title_id, sequences_min_hash) in enumerate(
            zip(train_data[c.COLUMN_TITLE_ID], train_data[c.COLUMN_SEQUENCES_MIN_HASH])):

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


def swap_word(x, length):
    words = x.split(' ')
    indexes = list(range(len(words)))
    replace_index = random.choice(indexes)
    to_replace_with_index = random.choice(indexes)
    words[replace_index], words[to_replace_with_index] = words[to_replace_with_index], words[replace_index]
    return ' '.join(words)


def generate_misspelled_name(word):
    new_word = str(word)
    functions = [random.choice([swap_word, add_letter, remove_letter]), replace_letter,
                 random.choice([add_space, remove_space])]
    functions_selected = random.sample(functions, random.randint(1, 2))
    for func in functions_selected:
        new_word = func(new_word, len(new_word))
    return new_word