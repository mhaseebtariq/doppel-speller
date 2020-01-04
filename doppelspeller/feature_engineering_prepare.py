import random
import time
import logging
import math

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.match_maker import MatchMaker


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


def get_closest_matches_per_training_row(train_data, truth_data):
    match_maker = MatchMaker(train_data, truth_data, s.TOP_N_RESULTS_TO_FIND_FOR_TRAINING)

    LOGGER.info('Preparing training features data!')

    closest_matches_per_training_row = {s.TRAIN_NOT_FOUND_VALUE: {}}
    number_training_rows = len(train_data)
    batch_time = time.time()
    for row_number, title_id in enumerate(train_data[c.COLUMN_TITLE_ID]):

        matches = match_maker.get_closest_matches(row_number)

        if title_id == s.TRAIN_NOT_FOUND_VALUE:
            closest_matches_per_training_row[title_id][row_number] = matches
            continue
        else:
            closest_matches_per_training_row[title_id] = matches

        if title_id not in matches:
            if len(closest_matches_per_training_row[title_id]) == s.TOP_N_RESULTS_TO_FIND_FOR_TRAINING:
                closest_matches_per_training_row[title_id].pop()

            closest_matches_per_training_row[title_id].append(title_id)

        if not(row_number % 5000):
            elapsed = f'{round(time.time() - batch_time)} secs'
            LOGGER.info(f'Processed {row_number} of {number_training_rows} [{elapsed}]!')
            batch_time = time.time()

    return closest_matches_per_training_row


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
    words = x.split()
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
