import json
import sqlite3
import logging
import time
import _pickle as pickle
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import xgboost as xgb

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import get_ground_truth, get_test_data, wait_for_multiprocessing_threads
from doppelspeller.feature_engineering import construct_features, get_ground_truth_words_counter


LOGGER = logging.getLogger(__name__)
(
    CONNECTION, CURSOR, LSH_FOREST, GROUND_TRUTH, GROUND_TRUTH_MAPPING_REVERSED, GROUND_TRUTH_MAPPING, MODEL,
    WORDS_COUNTER,NUMBER_OF_WORDS
) = (
    None, None, None, None, None, None, None, None, None
)


def populate_required_data():
    global LSH_FOREST, MODEL, GROUND_TRUTH, GROUND_TRUTH_MAPPING, GROUND_TRUTH_MAPPING_REVERSED, CONNECTION, CURSOR, \
        WORDS_COUNTER, NUMBER_OF_WORDS

    LOGGER.info('Reading LSH forest dump!')
    with open(s.LSH_FOREST_OUTPUT_FILE, 'rb') as fl:
        LSH_FOREST = pickle.load(fl)

    GROUND_TRUTH = get_ground_truth()
    GROUND_TRUTH.set_index(c.COLUMN_TITLE_ID, inplace=True)
    GROUND_TRUTH.sort_index(inplace=True)
    GROUND_TRUTH.loc[:, c.COLUMN_TITLE_ID] = GROUND_TRUTH.index

    CONNECTION = sqlite3.connect(s.SQLITE_DB)
    CURSOR = CONNECTION.cursor()

    GROUND_TRUTH_MAPPING = GROUND_TRUTH.set_index(c.COLUMN_TITLE_ID).copy(deep=True)
    GROUND_TRUTH_MAPPING = GROUND_TRUTH_MAPPING.to_dict()[c.COLUMN_TRANSFORMED_TITLE]
    GROUND_TRUTH_MAPPING = {
        k: {
            'truth_title': v,
            'truth_title_words': v.split(' '),
            'truth_number_of_characters': len(v),
            'truth_number_of_words': len(v.split(' ')),
        } for k, v in GROUND_TRUTH_MAPPING.items()
    }
    GROUND_TRUTH_MAPPING_REVERSED = {v['truth_title']: k for k, v in GROUND_TRUTH_MAPPING.items()}

    with open(s.MODEL_DUMP_FILE, 'rb') as fl:
        MODEL = pickle.load(fl)

    WORDS_COUNTER = get_ground_truth_words_counter(GROUND_TRUTH)
    NUMBER_OF_WORDS = len(WORDS_COUNTER)

    return True


def get_features_names():
    evaluation_set = pd.read_pickle(s.EVALUATION_OUTPUT_FILE)
    return list(evaluation_set.columns)


def get_nearest_matches(test_id):
    query = f'SELECT matches from {s.SQLITE_NEIGHBOURS_TABLE} WHERE test_id={test_id}'
    CURSOR.execute(query)
    return json.loads(CURSOR.fetchone()[0])


def save_prediction(test_id, title_to_match, best_match, best_match_id, best_match_probability):
    query = f"INSERT INTO {s.SQLITE_PREDICTIONS_TABLE} VALUES ({test_id}, '{title_to_match}', " \
            f"'{best_match}', {best_match_id}, {best_match_probability});"
    CURSOR.execute(query)
    CONNECTION.commit()
    return


def generate_single_prediction(index, title_to_match):
    title_to_match_words = title_to_match.split(' ')
    all_nearest_found = get_nearest_matches(index)

    for matches_nearest in [all_nearest_found[0:10], all_nearest_found[10:100], all_nearest_found[100:]]:
        if not matches_nearest:
            continue

        matches = [GROUND_TRUTH_MAPPING[x] for x in matches_nearest]
        len_matches = len(matches)

        truth_title = [x['truth_title'] for x in matches]
        truth_number_of_words = [x['truth_number_of_words'] for x in matches]
        truth_title_words = [x['truth_title_words'] for x in matches]
        truth_number_of_characters = [x['truth_number_of_characters'] for x in matches]

        title = [title_to_match] * len_matches
        number_of_characters = [len(title_to_match)] * len_matches
        number_of_words = [len(title_to_match_words)] * len_matches
        distance = list(map(lambda x: fuzz.ratio(x[0], x[1]), zip(truth_title, title)))

        extra_features = list(map(lambda x: construct_features(x[0], x[1], x[2], WORDS_COUNTER, NUMBER_OF_WORDS), zip(
            truth_number_of_words,
            truth_title_words,
            title)))

        features = np.array([
            number_of_characters,
            truth_number_of_characters,
            number_of_words,
            truth_number_of_words,
            distance
        ])

        features = np.concatenate([features.T, extra_features], axis=1)

        d_test = xgb.DMatrix(features, feature_names=get_features_names())
        predictions = MODEL.predict(d_test)
        best_match_index = np.argmax(predictions)
        best_match = matches[best_match_index]['truth_title']
        best_match_id = matches_nearest[best_match_index]
        best_match_prediction = predictions[best_match_index]

        if best_match_prediction > 0.5:
            save_prediction(index, title_to_match, best_match, best_match_id, best_match_prediction)
            break

    return


def generate_predictions():
    _ = populate_required_data()
    test_data = get_test_data()

    # Drop table
    CURSOR.execute(f"DROP TABLE IF EXISTS {s.SQLITE_PREDICTIONS_TABLE};")

    # Create table
    CURSOR.execute(f"CREATE TABLE {s.SQLITE_PREDICTIONS_TABLE} "
                   f"(test_id INTEGER, "
                   f"title_to_match TEXT, "
                   f"best_match TEXT, "
                   f"best_match_id INTEGER, "
                   f"best_match_probability FLOAT);")

    # Save (commit) the changes
    CONNECTION.commit()

    matched_so_far = []

    # 1st Step
    LOGGER.info('Finding exact matches!')

    test_data.loc[:, 'exact'] = test_data.loc[:, c.COLUMN_TRANSFORMED_TITLE].apply(
        lambda x: GROUND_TRUTH_MAPPING_REVERSED.get(x, -2))

    for index, row in test_data.loc[test_data['exact'] != -2, :].iterrows():
        title_to_match = row[c.COLUMN_TRANSFORMED_TITLE]
        best_match_id = row['exact']
        best_match = GROUND_TRUTH_MAPPING[best_match_id]['truth_title']

        matched_so_far.append(index)
        save_prediction(index, title_to_match, best_match, best_match_id, 1.0)

    # 2nd Step
    LOGGER.info('Finding very close matches!')

    remaining = test_data.loc[~test_data.index.isin(matched_so_far), :]
    remaining_count = len(remaining)
    count = 0
    for index, row in remaining.iterrows():
        count += 1
        if not (count % 10000):
            LOGGER.info(f'Processed {count} of {remaining_count}...')

        title_to_match = row[c.COLUMN_TRANSFORMED_TITLE]
        best_match_ids = get_nearest_matches(index)

        if not best_match_ids:
            matched_so_far.append(index)
            save_prediction(index, title_to_match, None, s.TRAIN_NOT_FOUND_VALUE, 0.0)
            continue

        matches = [GROUND_TRUTH_MAPPING[best_match_id]['truth_title'] for best_match_id in best_match_ids]
        ratios = [fuzz.ratio(best_match, title_to_match) for best_match in matches]
        arg_max = np.argmax(ratios)
        max_ratio = ratios[arg_max]
        best_match_id = best_match_ids[arg_max]
        if max_ratio > 94:
            matched_so_far.append(index)
            best_match = matches[arg_max]
            save_prediction(index, title_to_match, best_match, best_match_id, 1.0)
        else:
            ratios = [fuzz.token_sort_ratio(best_match, title_to_match) for best_match in matches]
            arg_max = np.argmax(ratios)
            max_ratio = ratios[arg_max]
            best_match_id = best_match_ids[arg_max]
            if max_ratio > 94:
                matched_so_far.append(index)
                best_match = matches[arg_max]
                save_prediction(index, title_to_match, best_match, best_match_id, 1.0)

    # 3rd Step
    LOGGER.info('Finding matches using the model!')

    executor = ProcessPoolExecutor(max_workers=3)
    threads = [
        executor.submit(generate_single_prediction, index, row[c.COLUMN_TRANSFORMED_TITLE])
        for index, row in test_data.loc[~test_data.index.isin(matched_so_far), :].iterrows()
    ]
    wait_for_multiprocessing_threads(threads)

    submission = pd.read_sql(
        f"SELECT test_id AS test_index, best_match_id AS {c.COLUMN_TITLE_ID} "
        f"FROM {s.SQLITE_PREDICTIONS_TABLE}", CONNECTION)

    not_matched_rows = test_data.loc[~test_data.index.isin(matched_so_far), :].copy(deep=True)
    not_matched_rows.index.name = 'test_index'
    not_matched_rows.reset_index(inplace=True)
    not_matched_rows.loc[:, c.COLUMN_TITLE_ID] = s.TRAIN_NOT_FOUND_VALUE
    not_matched_rows = not_matched_rows.loc[:, ['test_index', c.COLUMN_TITLE_ID]].copy(deep=True)

    final_submission = pd.concat([not_matched_rows, submission])
    final_submission.sort_values(['test_index'], inplace=True)
    final_submission.reset_index(drop=True, inplace=True)

    final_submission.to_csv(s.FINAL_OUTPUT_FILE, index=False, sep='|')
