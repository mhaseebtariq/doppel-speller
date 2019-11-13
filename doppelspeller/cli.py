import sys
import logging

import click

from doppelspeller import __version__, __build__
from doppelspeller.common import get_ground_truth


LOGGER = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__)
@click.option('-v', '--verbose', count=True, envvar='LOGGING_LEVEL',
              help='Make output more verbose. Use more v\'s for more verbose')
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
def generate_lsh_forest(**kwargs):
    """Generate the LSH forest for clustering the titles together!"""
    from doppelspeller.clustering import generate_lsh_forest

    return generate_lsh_forest(get_ground_truth())
