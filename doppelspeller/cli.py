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

    return generate_lsh_forest(get_ground_truth())


@cli.command()
@time_usage
def prepare_training(**kwargs):
    """Prepare data for training the model!"""
    from doppelspeller.train_preparation import train_preparation, generate_dummy_train_data

    LOGGER.info('Preparing train data!')
    _ = train_preparation()
    _ = generate_dummy_train_data()

    return


@cli.command()
@time_usage
def train_model(**kwargs):
    """Train the model!"""
    from doppelspeller.train import train_model

    return train_model()
