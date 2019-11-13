import _pickle as pickle
from datetime import datetime

import pandas as pd
import xgboost as xgb
from fuzzywuzzy import fuzz

SEQUENCES_OF = 3
NUM_PERM = 128


def train():
    in_top = 10
    with open('similar_programs_{}.dump'.format(in_top), 'rb') as fl:
        lsh_forest_data = pickle.load(fl)
    training_data_negative = lsh_forest_data.pop(-1)

    ground_truth_mapping = ground_truth.set_index('company_id').copy(deep=True)
    ground_truth_mapping = ground_truth_mapping.to_dict()['transformed_name']

    train_data = pd.read_csv('STrain.csv', delimiter='|')
    train_data.loc[:, 'transformed_name'] = train_data.loc[:, 'name'].apply(lambda x: convert_text(x))
    train_data = train_data.set_index('company_id')
    del train_data['name']
    train_data_mapping = train_data.to_dict()['transformed_name']

    train_data_negatives_mapping = train_data[train_data.index == -1].copy(deep=True)
    train_data_negatives_mapping = train_data_negatives_mapping.set_index('train_index').to_dict()['transformed_name']

    generated_training_data = pd.read_pickle('generated_training_data.dump')

    generated_training_data.head()

    training_rows_generated = []
    for train_index, row in generated_training_data.iterrows():
        truth_company_name = row['transformed_name']
        company_name = row['generated_misspelled']
        training_rows_generated.append(
            [1, company_name, company_name.split(' '), truth_company_name, truth_company_name.split(' '), 1])

    training_rows_negative = []
    for train_index, companies in training_data_negative.items():
        company_name = train_data_negatives_mapping[train_index]
        for truth_company_id in companies:
            truth_company_name = ground_truth_mapping[truth_company_id]
            truth_company_name_words = truth_company_name.split(' ')
            company_name_words = company_name.split(' ')
            training_rows_negative.append(
                [2, company_name, company_name_words, truth_company_name, truth_company_name_words, 0])

    training_rows = []
    for company_id, companies in lsh_forest_data.items():
        company_name = train_data_mapping[company_id]
        for truth_company_id in companies:
            truth_company_name = ground_truth_mapping[truth_company_id]
            truth_company_name_words = truth_company_name.split(' ')
            company_name_words = company_name.split(' ')
            training_rows.append(
                [3, company_name, company_name_words, truth_company_name, truth_company_name_words,
                 int(company_id == truth_company_id)])

    training_rows_final = training_rows_negative + training_rows + training_rows_generated

    train = pd.DataFrame(training_rows_final, columns=[
        'kind', 'company_name', 'company_name_words', 'truth_company_name', 'truth_company_name_words', 'target'])

    train.loc[:, 'number_of_characters'] = train.loc[:, 'company_name'].apply(
        lambda x: len(x)
    )
    train.loc[:, 'truth_number_of_characters'] = train.loc[:, 'truth_company_name'].apply(
        lambda x: len(x)
    )
    train.loc[:, 'number_of_words'] = train.loc[:, 'company_name_words'].apply(
        lambda x: len(x)
    )
    train.loc[:, 'truth_number_of_words'] = train.loc[:, 'truth_company_name_words'].apply(
        lambda x: len(x)
    )
    train.loc[:, 'levenshtein'] = list(
        map(lambda x: fuzz.ratio(x[0], x[1]), zip(train.loc[:, 'company_name'], train.loc[:, 'truth_company_name'])))

    train.head()

    len(train)

    # Takes ~9 minutes

    st = time.time()

    extra_features = list(map(lambda x: construct_features(x[0], x[1], x[2]), zip(
        train.loc[:, 'truth_number_of_words'],
        train.loc[:, 'truth_company_name_words'],
        train.loc[:, 'company_name'])))

    print(round((time.time() - st) / 60, 2))

    columns = ['truth_{}th_word_length'.format(x + 1) for x in range(15)]
    columns += ['truth_{}th_word_probability'.format(x + 1) for x in range(15)]
    columns += ['truth_{}th_word_probability_rank'.format(x + 1) for x in range(15)]
    columns += ['truth_{}th_word_best_match_score'.format(x + 1) for x in range(15)]
    columns.append('reconstructed_score')

    extra = pd.DataFrame(extra_features, columns=columns)

    train = train.merge(extra, right_index=True, left_index=True)

    evaluation_generated = train.loc[train['kind'] == 1, :].sample(frac=0.05).copy(deep=True)
    evalutaion_negative = train.loc[train['kind'] == 2, :].sample(frac=0.1).copy(deep=True)
    evaluation_positive = train.loc[train['kind'] == 3, :].sample(frac=0.05).copy(deep=True)

    evaluation = pd.concat([evaluation_generated, evalutaion_negative, evaluation_positive])

    evaluation_indexes = list(evaluation_generated.index) + list(evalutaion_negative.index) + list(
        evaluation_positive.index)

    train = train.loc[~train.index.isin(evaluation_indexes), :].copy(deep=True).reset_index(drop=True)

    evaluation = evaluation.reset_index(drop=True)

    del train['kind']
    del evaluation['kind']

    len(train), len(evaluation)

    # train.to_pickle('train.dump')
    # evaluation.to_pickle('evaluation.dump')


    # train = pd.read_pickle('train.dump')
    # evaluation = pd.read_pickle('evaluation.dump')


    train_set = train.loc[:, ~train.columns.isin(
        ['company_name', 'company_name_words', 'truth_company_name', 'truth_company_name_words', 'target'])]
    train_set_target = train.loc[:, 'target']

    d_train = xgb.DMatrix(train_set.values, label=train_set_target, feature_names=train_set.columns)

    evaluation_set = evaluation.loc[:, ~evaluation.columns.isin(
        ['company_name', 'company_name_words', 'truth_company_name', 'truth_company_name_words', 'target'])]
    evaluation_set_target = evaluation.loc[:, 'target']

    d_evaluation = xgb.DMatrix(evaluation_set.values, label=evaluation_set_target, feature_names=evaluation_set.columns)

    num_rows_train = len(train_set)


    def custom_error(predictions, train_or_eval):
        actuals = train_or_eval.get_label()
        is_train_set = False
        if len(actuals) == num_rows_train:
            is_train_set = True

        threshold = 0.5

        predictions_negative_indexes = (predictions < threshold).nonzero()[0]
        predictions_positive_indexes = (predictions >= threshold).nonzero()[0]

        false_negative_cost = sum(actuals[predictions_negative_indexes])
        false_positive_cost = sum(actuals[predictions_positive_indexes] == 0) * 5

        cost = false_negative_cost + false_positive_cost

        return 'custom-error', cost


    def weighted_logloss(predictions, train):
        beta = 5
        actuals = train.get_label()
        gradient = predictions * (beta + actuals - beta * actuals) - actuals
        hessian = predictions * (1 - predictions) * (beta + actuals - beta * actuals)
        return gradient, hessian


    scale_pos_weight = sum(train_set_target == 0) / sum(train_set_target == 1)
    print(scale_pos_weight)

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
        obj=weighted_logloss,
        maximize=False,
        **params
    )


    def get_xgb_feats_importance(reg):
        fscore = reg.get_fscore()

        feat_importances = []
        for ft, score in fscore.items():
            feat_importances.append({'feature': ft, 'importance': score})

        feat_importances = pd.DataFrame(feat_importances)
        feat_importances = feat_importances.sort_values(by='importance', ascending=False).reset_index(drop=True)
        feat_importances['importance'] /= feat_importances['importance'].sum()
        return feat_importances


    features_importance = get_xgb_feats_importance(model)

    ax = features_importance.head(20).plot.barh(x='feature', y='importance')

    sum(features_importance.head(20)['importance'])

    postfix = str(datetime.now()).replace(' ', '').replace(':', '-').replace('.', '-')
    with open('model-{}.dump'.format(postfix), 'wb') as fl:
        pickle.dump(model, fl)
