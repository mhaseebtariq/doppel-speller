import sys
import os
import logging

import click

from doppelspeller import __version__, __build__
from doppelspeller.cli_utils import time_usage


LOGGER = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__)
@click.option('-v', '--verbose', count=True, envvar='LOGGING_LEVEL',
              help='Make output more verbose. Use more v\'s for more verbosity.')
@click.pass_context
def cli(context, verbose):
    LOGGER.info(f"Predict Redeem v{__version__}-{__build__}")

    if verbose <= 1:
        level = logging.WARNING
    elif verbose == 2:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(stream=sys.stdout, level=level)


@cli.command()
def stage_example_data_set_on_docker_container(**kwargs):
    """Makes the example data set files available on the Docker container!"""
    cmd = (
        "cp -r /doppelspeller/example_dataset/*.gz $PROJECT_DATA_PATH && "
        "cd $PROJECT_DATA_PATH && /bin/gunzip -f -r *.gz"
    )
    return os.popen(cmd).read()


@cli.command()
@time_usage
def pre_process_data(**kwargs):
    """Prepare data sets!"""
    import _pickle as pickle

    import doppelspeller.settings as s
    import doppelspeller.constants as c
    from doppelspeller.common import get_ground_truth, get_train_data, get_test_data
    from doppelspeller.match_maker import MatchMaker

    ground_truth, train_data, test_data = get_ground_truth(), get_train_data(), get_test_data()

    LOGGER.info(f'Finding nearest {s.TOP_N_RESULTS_TO_FIND_FOR_TRAINING} matches for "train" data!')
    matcher_train = MatchMaker(train_data, ground_truth,
                               s.TOP_N_RESULTS_TO_FIND_FOR_TRAINING)
    matcher_train.process()
    matches_train = matcher_train.closest_matches
    del matcher_train

    LOGGER.info(f'Finding nearest {s.TOP_N_RESULTS_TO_FIND_FOR_PREDICTING} matches for "test" data!')
    matcher_test = MatchMaker(test_data, ground_truth, s.TOP_N_RESULTS_TO_FIND_FOR_PREDICTING)
    matcher_test.process()
    matches_test = matcher_test.closest_matches
    del matcher_test

    with open(s.PRE_REQUISITE_TRAIN_DATA_FILE, 'wb') as fl:
        pickle.dump({
            c.DATA_TYPE_TRUTH: ground_truth,
            c.DATA_TYPE_TRAIN: train_data,
            c.DATA_TYPE_NEAREST_TRAIN: matches_train
        }, fl)
    with open(s.PRE_REQUISITE_TEST_DATA_FILE, 'wb') as fl:
        pickle.dump({
            c.DATA_TYPE_TRUTH: ground_truth,
            c.DATA_TYPE_TEST: test_data,
            c.DATA_TYPE_NEAREST_TEST: matches_test,
        }, fl)

    return True


@cli.command()
@time_usage
def train_model(**kwargs):
    """Train the model!"""
    from doppelspeller.train import train_model

    LOGGER.info('Training the model!')
    return train_model()


@cli.command()
@time_usage
def generate_predictions(**kwargs):
    """Generate the predictions!"""
    from doppelspeller.predict import Prediction

    LOGGER.info('Generating the predictions!')
    prediction = Prediction()
    return prediction.process()


@cli.command()
@click.option('-t', '--title-to-search', 'title')
@time_usage
def extensive_search_single_title(**kwargs):
    """Extensive search single title!"""
    from doppelspeller.predict import Prediction

    LOGGER.info('Searching for the closest match!')

    title_to_search = kwargs['title'].strip()
    if not title_to_search:
        raise Exception('Empty value provided for --title-to-search="" (direct call) or title="" (make call)')

    prediction = Prediction()
    found = prediction.extensive_search_single_title(title_to_search)

    LOGGER.info(f'Title: {kwargs["title"]}')
    LOGGER.info(f'Closest match: {found}')
    return found


@cli.command()
@time_usage
def get_predictions_accuracy(**kwargs):
    """Print predictions accuracy!"""
    import pandas as pd

    import doppelspeller.constants as c
    import doppelspeller.settings as s

    try:
        actual = pd.read_csv(
            s.TEST_WITH_ACTUALS_FILE, sep=s.TEST_FILE_DELIMITER).to_dict()[s.TEST_WITH_ACTUALS_TITLE_ID]
    except:  # noqa
        raise Exception(f'Error reading {s.TEST_WITH_ACTUALS_FILE} (TEST_WITH_ACTUAL_FILE in settings.py)')
    predictions = pd.read_csv(s.FINAL_OUTPUT_FILE, sep=s.TEST_FILE_DELIMITER).to_dict()[c.COLUMN_TITLE_ID]

    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0
    for key, value in actual.items():
        value_output = predictions[key]
        if value == -1:
            if value_output == -1:
                true_negatives += 1
            else:
                false_negatives += 1
        else:
            if value_output == value:
                true_positives += 1
            else:
                false_positives += 1

    LOGGER.info(f"""\n
    True Positives          {true_positives}
    True Negatives          {true_negatives}
    False Positives         {false_positives}
    False Negatives         {false_negatives}
    """)

    return true_positives, true_negatives, false_positives, false_negatives
