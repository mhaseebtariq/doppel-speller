import timeit
import logging
from datetime import timedelta
from functools import wraps


LOGGER = logging.getLogger(__name__)


def format_time(seconds):
    formatted = str(timedelta(seconds=seconds)).split(':')
    return f"{formatted[0]} hours | {int(formatted[1])} minutes | {int(formatted[2].split('.')[0])} seconds"


def time_usage(func):  # pragma: no cover
    """
    Decorator to time a function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        return_value = func(*args, **kwargs)
        elapsed = timeit.default_timer() - start_time
        LOGGER.info(f"function[={func.__name__}] executed in {format_time(elapsed)}!")
        return return_value

    return wrapper
