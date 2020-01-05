import logging
import _pickle as pickle

import numba
import pandas as pd
import numpy as np
import xgboost as xgb

import doppelspeller.settings as s
import doppelspeller.constants as c
from doppelspeller.feature_engineering import FeatureEngineering


LOGGER = logging.getLogger(__name__)


@numba.njit(fastmath=True)
def fast_custom_error(prediction, actual):
    predictions_negative_indexes = (prediction <= s.PREDICTION_PROBABILITY_THRESHOLD).nonzero()[0]
    predictions_positive_indexes = (prediction > s.PREDICTION_PROBABILITY_THRESHOLD).nonzero()[0]

    false_negative_cost = np.sum(actual[predictions_negative_indexes])
    false_positive_cost = np.sum(actual[predictions_positive_indexes] == 0) * s.FALSE_POSITIVE_PENALTY_FACTOR

    return false_negative_cost + false_positive_cost


@numba.njit(fastmath=True)
def fast_weighted_log_loss(prediction, actual):
    beta = s.FALSE_POSITIVE_PENALTY_FACTOR
    gradient = prediction * (beta + actual - beta * actual) - actual
    hessian = prediction * (1 - prediction) * (beta + actual - beta * actual)
    return gradient, hessian


def custom_error(predictions, train_data_object):
    return 'custom-error', fast_custom_error(predictions, train_data_object.get_label())


def weighted_log_loss(predictions, train_data_object):
    return fast_weighted_log_loss(predictions, train_data_object.get_label())


def get_xgb_feats_importance(model):
    features_score = model.get_fscore()

    features_importance = []
    for feature, score in features_score.items():
        features_importance.append({'feature': feature, 'importance': score})

    features_importance = pd.DataFrame(features_importance)
    features_importance = features_importance.sort_values(by='importance', ascending=False).reset_index(drop=True)
    features_importance['importance'] /= features_importance['importance'].sum()
    return features_importance


def train_model():
    LOGGER.info('Generating train and evaluation data-sets!')

    features = FeatureEngineering(c.DATA_TYPE_TRAIN)
    train, train_target, evaluation, evaluation_target = features.generate_train_and_evaluation_data_sets()

    train_set = np.array(train.tolist(), dtype=np.float16)
    del train

    evaluation_set = np.array(evaluation.tolist(), dtype=np.float16)
    del evaluation

    d_train = xgb.DMatrix(train_set, label=train_target)
    d_evaluation = xgb.DMatrix(evaluation_set, label=evaluation_target)

    scale_pos_weight = sum(train_target == 0) / sum(train_target == 1)

    watch_list = [(d_train, 'train'), (d_evaluation, 'evaluation')]
    params = {
        'params': {
            'max_depth': 5,
            'eta': 0.1,
            'nthread': 4,  # TODO: Use dynamic settings
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

    features_importance_data = get_xgb_feats_importance(model)

    with open(s.MODEL_DUMP_FILE, 'wb') as file_object:
        pickle.dump(model, file_object)

    return features_importance_data
