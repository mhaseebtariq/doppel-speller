import sys
import logging

import click

from doppelspeller import __version__, __build__
from doppelspeller.common import get_ground_truth
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
@time_usage
def generate_lsh_forest(**kwargs):
    """Generate the LSH forest for clustering the titles together!"""
    from doppelspeller.clustering import generate_lsh_forest

    LOGGER.info('Generating LSH forest!')
    return generate_lsh_forest(get_ground_truth())


@cli.command()
@time_usage
def prepare_data_for_features_generation(**kwargs):
    """Prepare data for features generation!"""
    from doppelspeller.feature_engineering import prepare_data_for_features_generation

    LOGGER.info('Preparing data for features generation!')
    return prepare_data_for_features_generation()


@cli.command()
@time_usage
def generate_train_and_evaluation_data_sets(**kwargs):
    """Generate train and evaluation data-sets!"""
    from doppelspeller.feature_engineering import generate_train_and_evaluation_data_sets

    LOGGER.info('Generating train and evaluation data-sets!')
    return generate_train_and_evaluation_data_sets()


@cli.command()
@time_usage
def train_model(**kwargs):
    """Train the model!"""
    from doppelspeller.train import train_model

    LOGGER.info('Training the model!')
    return train_model()


@cli.command()
@time_usage
def prepare_predictions_data(**kwargs):
    """Prepare the predictions data required for generating the predictions!"""
    from doppelspeller.predict_preparation import PrePredictionData

    LOGGER.info('Preparing the predictions data!')
    pre_prediction_data = PrePredictionData()
    return pre_prediction_data.process()


@cli.command()
@time_usage
def generate_predictions(**kwargs):
    """Generate the predictions!"""
    from doppelspeller.predict import Prediction

    LOGGER.info('Generating the predictions!')
    prediction = Prediction()
    return prediction.process()


@cli.command()
@click.option('-t', '--title-to-search', 'title', default='U V GRpOUP PLC')
@time_usage
def extensive_search_single_title(**kwargs):
    """Extensive search single title!"""
    from doppelspeller.predict import Prediction

    LOGGER.info('Searching for the closest match!')

    prediction = Prediction()
    found = prediction.extensive_search_single_title(kwargs['title'])

    LOGGER.info(f'Title: {kwargs["title"]}')
    LOGGER.info(f'Closest match: {found}')
    return found
