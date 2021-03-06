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

    logging.basicConfig(stream=sys.stdout, level=level, format='[%(asctime)s]%(levelname)s|%(name)s|%(message)s')


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
def train_model(**kwargs):
    """Train the model!"""
    from doppelspeller.train import train_model

    LOGGER.info('Training the model!')
    return train_model()


@cli.command()
@time_usage
def generate_predictions(**kwargs):
    """Generate the predictions!"""
    import doppelspeller.constants as c
    from doppelspeller.predict import Prediction

    LOGGER.info('Generating the predictions!')
    prediction = Prediction(c.DATA_TYPE_TEST)
    return prediction.generate_test_predictions()


@cli.command()
@click.option('-t', '--title-to-search', 'title')
@time_usage
def closest_search_single_title(**kwargs):
    """Closest search single title!"""
    import doppelspeller.constants as c
    from doppelspeller.predict import Prediction

    LOGGER.info('Searching for the closest match!')

    title_to_search = kwargs['title'].strip()
    if not title_to_search:
        raise Exception('Empty value provided for --title-to-search="" (direct call) or title="" (make call)')

    prediction = Prediction(c.DATA_TYPE_SINGLE, title=title_to_search)
    found = prediction.generate_test_predictions(single_prediction=True)

    LOGGER.info(f'Title: {kwargs["title"]}')
    LOGGER.info(f'\n\nClosest match: {found}\n')
    return found


@cli.command()
@time_usage
def get_predictions_accuracy(**kwargs):
    """Print predictions accuracy!"""
    import pandas as pd

    import doppelspeller.constants as c
    import doppelspeller.settings as s

    try:
        actual = pd.read_csv(s.TEST_WITH_ACTUALS_FILE, sep=s.TEST_FILE_DELIMITER)
    except:  # noqa
        raise Exception(f'Error reading {s.TEST_WITH_ACTUALS_FILE} (TEST_WITH_ACTUAL_FILE in settings.py)')
    predictions = pd.read_csv(s.FINAL_OUTPUT_FILE, sep=s.TEST_FILE_DELIMITER)

    actual.set_index(c.COLUMN_TEST_INDEX, inplace=True)
    predictions.set_index(c.COLUMN_TEST_INDEX, inplace=True)

    actual = actual.to_dict()[s.TEST_WITH_ACTUALS_TITLE_ID]
    predictions = predictions.to_dict()[c.COLUMN_TITLE_ID]

    correctly_matched_existing, correctly_matched_non_existing = 0, 0
    incorrectly_matched_existing, incorrectly_matched_non_existing = 0, 0
    for key, actual_value in actual.items():
        prediction_value = predictions[key]
        if prediction_value == -1:
            if actual_value == prediction_value:
                correctly_matched_non_existing += 1
            else:
                incorrectly_matched_non_existing += 1
        else:
            if actual_value == prediction_value:
                correctly_matched_existing += 1
            else:
                incorrectly_matched_existing += 1

    LOGGER.info(f"""\n
    Correctly matched titles            {correctly_matched_existing}
    Incorrectly matched titles          {incorrectly_matched_existing}
    Correctly marked as not-found       {correctly_matched_non_existing}
    Incorrectly marked as not-found     {incorrectly_matched_non_existing}

    Custom Error                        {incorrectly_matched_non_existing + (incorrectly_matched_existing * 5)}
    [custom_error = number_of_incorrectly_matched_non_existing + (number_of_incorrectly_matched_existing * 5)]
    """)

    return True
