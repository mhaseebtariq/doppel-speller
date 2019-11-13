import json
import _pickle as pickle
import sqlite3
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import xgboost as xgb

SEQUENCES_OF = 3
NUM_PERM = 128


def predict():
    with open('lsh_forest.dump', 'rb') as fl:
        lsh_forest = pickle.load(fl)

    ground_truth_mapping = ground_truth.set_index('company_id').copy(deep=True)
    ground_truth_mapping = ground_truth_mapping.to_dict()['transformed_name']
    ground_truth_mapping = {
        k: {
            'truth_company_name': v,
            'truth_company_name_words': v.split(' '),
            'truth_number_of_characters': len(v),
            'truth_number_of_words': len(v.split(' ')),
        } for k, v in ground_truth_mapping.items()
    }
    ground_truth_mapping_reversed = {v['truth_company_name']: k for k, v in ground_truth_mapping.items()}

    test_data = pd.read_csv('STest.csv', delimiter='|')
    test_data.loc[:, 'transformed_name'] = test_data.loc[:, 'name'].apply(lambda x: convert_text(x))
    del test_data['name']
    del test_data['test_index']
    test_data.loc[:, 'sequences'] = test_data.loc[:, 'transformed_name'].apply(
        lambda x: get_sequences(x, SEQUENCES_OF)
    )

    connection = sqlite3.connect('data.db')
    cursor = connection.cursor()

    # Drop table
    cursor.execute('''DROP TABLE IF EXISTS predictions;''')

    # Create table
    cursor.execute('''
        CREATE TABLE predictions (test_id INTEGER, 
                                  title_to_match TEXT, 
                                  best_match TEXT, 
                                  best_match_id INTEGER, 
                                  best_match_probability FLOAT);
    ''')

    # Save (commit) the changes
    connection.commit()


    def get_nearest_matches(test_id):
        query = 'SELECT matches from neighbours WHERE test_id={}'.format(test_id)
        cursor.execute(query)
        return json.loads(cursor.fetchone()[0])


    def save_prediction(test_id, title_to_match, best_match, best_match_id, best_match_probability):
        query = f"INSERT INTO predictions VALUES ({test_id}, '{title_to_match}', '{best_match}', {best_match_id}, {best_match_probability});"
        cursor.execute(query)
        connection.commit()
        return


    # 1st Step

    matched_so_far = []

    test_data.loc[:, 'exact'] = test_data.loc[:, 'transformed_name'].apply(
        lambda x: ground_truth_mapping_reversed.get(x, False))

    for index, row in test_data.loc[test_data['exact'] != False, :].iterrows():
        title_to_match = row['transformed_name']
        best_match_id = row['exact']
        best_match = ground_truth_mapping[best_match_id]['truth_company_name']

        matched_so_far.append(index)
        save_prediction(index, title_to_match, best_match, best_match_id, 1.0)

    len(test_data.loc[~test_data.index.isin(matched_so_far), :])

    # 2nd Step

    count = 0
    for index, row in test_data.loc[~test_data.index.isin(matched_so_far), :].iterrows():
        count += 1
        if not (count % 10000):
            print(count)

        title_to_match = row['transformed_name']
        best_match_ids = get_nearest_matches(index)

        if not best_match_ids:
            matched_so_far.append(index)
            save_prediction(index, title_to_match, None, -1, 0.0)
            continue

        matches = [ground_truth_mapping[best_match_id]['truth_company_name'] for best_match_id in best_match_ids]
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

    len(test_data.loc[~test_data.index.isin(matched_so_far), :])

    with open('model.dump', 'rb') as fl:
        model = pickle.load(fl)

    feature_names = [
        'number_of_characters', 'truth_number_of_characters', 'number_of_words',
        'truth_number_of_words', 'levenshtein', 'truth_1th_word_length',
        'truth_2th_word_length', 'truth_3th_word_length',
        'truth_4th_word_length', 'truth_5th_word_length',
        'truth_6th_word_length', 'truth_7th_word_length',
        'truth_8th_word_length', 'truth_9th_word_length',
        'truth_10th_word_length', 'truth_11th_word_length',
        'truth_12th_word_length', 'truth_13th_word_length',
        'truth_14th_word_length', 'truth_15th_word_length',
        'truth_1th_word_probability', 'truth_2th_word_probability',
        'truth_3th_word_probability', 'truth_4th_word_probability',
        'truth_5th_word_probability', 'truth_6th_word_probability',
        'truth_7th_word_probability', 'truth_8th_word_probability',
        'truth_9th_word_probability', 'truth_10th_word_probability',
        'truth_11th_word_probability', 'truth_12th_word_probability',
        'truth_13th_word_probability', 'truth_14th_word_probability',
        'truth_15th_word_probability', 'truth_1th_word_probability_rank',
        'truth_2th_word_probability_rank', 'truth_3th_word_probability_rank',
        'truth_4th_word_probability_rank', 'truth_5th_word_probability_rank',
        'truth_6th_word_probability_rank', 'truth_7th_word_probability_rank',
        'truth_8th_word_probability_rank', 'truth_9th_word_probability_rank',
        'truth_10th_word_probability_rank', 'truth_11th_word_probability_rank',
        'truth_12th_word_probability_rank', 'truth_13th_word_probability_rank',
        'truth_14th_word_probability_rank', 'truth_15th_word_probability_rank',
        'truth_1th_word_best_match_score', 'truth_2th_word_best_match_score',
        'truth_3th_word_best_match_score', 'truth_4th_word_best_match_score',
        'truth_5th_word_best_match_score', 'truth_6th_word_best_match_score',
        'truth_7th_word_best_match_score', 'truth_8th_word_best_match_score',
        'truth_9th_word_best_match_score', 'truth_10th_word_best_match_score',
        'truth_11th_word_best_match_score', 'truth_12th_word_best_match_score',
        'truth_13th_word_best_match_score', 'truth_14th_word_best_match_score',
        'truth_15th_word_best_match_score', 'reconstructed_score'
    ]


    def generate_prediction(index, title_to_match):
        title_to_match_words = title_to_match.split(' ')

        all_nearest_found = get_nearest_matches(index)

        for matches_nearest in [all_nearest_found[0:10], all_nearest_found[10:100], all_nearest_found[100:]]:

            matches = [ground_truth_mapping[x] for x in matches_nearest]
            len_matches = len(matches)

            truth_company_name = [x['truth_company_name'] for x in matches]
            truth_number_of_words = [x['truth_number_of_words'] for x in matches]
            truth_company_name_words = [x['truth_company_name_words'] for x in matches]
            truth_number_of_characters = [x['truth_number_of_characters'] for x in matches]

            company_name = [title_to_match] * len_matches
            company_name_words = [title_to_match_words] * len_matches
            number_of_characters = [len(title_to_match)] * len_matches
            number_of_words = [len(title_to_match_words)] * len_matches
            levenshtein = list(map(lambda x: fuzz.ratio(x[0], x[1]), zip(truth_company_name, company_name)))

            extra_features = list(map(lambda x: construct_features(x[0], x[1], x[2]), zip(
                truth_number_of_words,
                truth_company_name_words,
                company_name)))

            features = np.array([
                number_of_characters,
                truth_number_of_characters,
                number_of_words,
                truth_number_of_words,
                levenshtein
            ])

            features = np.concatenate([features.T, extra_features], axis=1)

            d_test = xgb.DMatrix(features, feature_names=feature_names)
            predictions = model.predict(d_test)
            best_match_index = np.argmax(predictions)
            best_match = matches[best_match_index]['truth_company_name']
            best_match_id = matches_nearest[best_match_index]
            best_match_prediction = predictions[best_match_index]

            if best_match_prediction > 0.5:
                matched_so_far.append(index)
                save_prediction(index, title_to_match, best_match, best_match_id, best_match_prediction)
                break

        return


    # 3rd Step

    executor = ProcessPoolExecutor(max_workers=3)
    threads = [
        executor.submit(generate_prediction, index, row['transformed_name'])
        for index, row in test_data.loc[~test_data.index.isin(matched_so_far), :].iterrows()
    ]

    running = sum([x.running() for x in threads])
    while running != 0:
        time.sleep(0.5)
        running = sum([x.running() for x in threads])

    len(test_data.loc[~test_data.index.isin(matched_so_far), :])

    submission = pd.read_sql("SELECT test_id AS test_index, best_match_id AS company_id FROM predictions", connection)

    not_macthed_rows = test_data.loc[~test_data.index.isin(matched_so_far), :].copy(deep=True)
    not_macthed_rows.index.name = 'test_index'
    not_macthed_rows.reset_index(inplace=True)
    not_macthed_rows.loc[:, 'company_id'] = -1
    not_macthed_rows = not_macthed_rows.loc[:, ['test_index', 'company_id']].copy(deep=True)

    final_submission = pd.concat([not_macthed_rows, submission])
    final_submission.sort_values(['test_index'], inplace=True)
    final_submission.reset_index(drop=True, inplace=True)

    final_submission.to_csv('final_submission.csv', index=False, sep='|')

    final_submission['company_id'].nunique(), len(final_submission[final_submission['company_id'] != -1])
