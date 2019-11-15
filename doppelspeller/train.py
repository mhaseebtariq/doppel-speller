import logging
import _pickle as pickle

import pandas as pd
import xgboost as xgb
from fuzzywuzzy import fuzz

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.common import get_ground_truth, get_train_data, construct_features, get_ground_truth_words_counter


LOGGER = logging.getLogger(__name__)


def train_model():
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
    train.loc[:, 'levenshtein'] = list(
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

    d_train = xgb.DMatrix(train_set.values, label=train_set_target, feature_names=train_set.columns)

    evaluation_set = evaluation.loc[:, ~evaluation.columns.isin(
        ['title', 'title_words', 'truth_title', 'truth_title_words', 'target'])]
    evaluation_set_target = evaluation.loc[:, 'target']

    d_evaluation = xgb.DMatrix(evaluation_set.values, label=evaluation_set_target, feature_names=evaluation_set.columns)

    num_rows_train = len(train_set)

    def custom_error(predictions, train_or_eval):
        actual_target = train_or_eval.get_label()
        is_train_set = False
        if len(actual_target) == num_rows_train:
            is_train_set = True

        threshold = 0.5

        predictions_negative_indexes = (predictions < threshold).nonzero()[0]
        predictions_positive_indexes = (predictions >= threshold).nonzero()[0]

        false_negative_cost = sum(actual_target[predictions_negative_indexes])
        false_positive_cost = sum(actual_target[predictions_positive_indexes] == 0) * 5

        cost = false_negative_cost + false_positive_cost

        return 'custom-error', cost

    def weighted_log_loss(predictions, train_data_object):
        beta = 5
        actual_target = train_data_object.get_label()
        gradient = predictions * (beta + actual_target - beta * actual_target) - actual_target
        hessian = predictions * (1 - predictions) * (beta + actual_target - beta * actual_target)
        return gradient, hessian

    scale_pos_weight = sum(train_set_target == 0) / sum(train_set_target == 1)

    watch_list = [(d_train, 'train'), (d_evaluation, 'evaluation')]
    params = {
        'params': {
            'max_depth': 5,
            'eta': 0.1,
            'nthread': 4,
            'min_child_weight': 1,
            'eval_metric': 'auc',
            'objective': 'reg:logistic',
            'scale_pos_weight': scale_pos_weight,
            'subsample': 1,
        },
        'num_boost_round': 1000,
        'verbose_eval': True,
        'early_stopping_rounds': 50,
    }

    model = xgb.train(
        dtrain=d_train,
        evals=watch_list,
        feval=custom_error,
        obj=weighted_log_loss,
        maximize=False,
        **params
    )

    def get_xgb_feats_importance(reg):
        features_score = reg.get_fscore()

        features_importance = []
        for feature, score in features_score.items():
            features_importance.append({'feature': feature, 'importance': score})

        features_importance = pd.DataFrame(features_importance)
        features_importance = features_importance.sort_values(by='importance', ascending=False).reset_index(drop=True)
        features_importance['importance'] /= features_importance['importance'].sum()
        return features_importance

    features_importance_data = get_xgb_feats_importance(model)

    with open(s.MODEL_DUMP_FILE, 'wb') as fl:
        pickle.dump(model, fl)

    return features_importance_data
