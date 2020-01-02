__version__ = '0.1.0'
__build__ = 'dev'
module_name = 'doppelspeller'

try:
    from numba import njit
    from numba.typed import List

    from doppelspeller.settings import DISABLE_NUMBA  # noqa
except ImportError:
    DISABLE_NUMBA = True


def dummy_njit(*args, **kwargs):
    def inner(func):
        return func
    return inner


if DISABLE_NUMBA:
    import logging

    LOGGER = logging.getLogger(__name__)
    LOGGER.warning('\nNumba (http://numba.pydata.org/) could not be imported; or is disabled. '
                   'Enable/install for faster executions!\n')
    List = list  # noqa
    njit = dummy_njit  # noqa
