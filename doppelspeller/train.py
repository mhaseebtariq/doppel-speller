import logging
import _pickle as pickle

import pandas as pd
import xgboost as xgb

import doppelspeller.settings as s


LOGGER = logging.getLogger(__name__)


def custom_error(predictions, train_or_evaluation):
    threshold = 0.5

    actual_target = train_or_evaluation.get_label()

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
    train_set = pd.read_pickle(s.TRAIN_OUTPUT_FILE)
    train_set_target = pd.read_pickle(s.TRAIN_TARGET_OUTPUT_FILE)
    evaluation_set = pd.read_pickle(s.EVALUATION_OUTPUT_FILE)
    evaluation_set_target = pd.read_pickle(s.EVALUATION_TARGET_OUTPUT_FILE)

    d_train = xgb.DMatrix(train_set.values, label=train_set_target, feature_names=train_set.columns)
    d_evaluation = xgb.DMatrix(evaluation_set.values, label=evaluation_set_target, feature_names=evaluation_set.columns)

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

    features_importance_data = get_xgb_feats_importance(model)

    with open(s.MODEL_DUMP_FILE, 'wb') as file_object:
        pickle.dump(model, file_object, protocol=s.PICKLE_PROTOCOL)

    return features_importance_data
