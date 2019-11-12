from setuptools import setup, find_packages

from doppelspeller import __version__, module_name

setup(
    name='doppel-speller',
    version=__version__,
    description='DoppelSpeller',
    author='Haseeb Tariq',
    author_email='mhaseebtariq@gmail.com',
    url='https://github.com/mhaseebtariq/doppel-speller',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=[
        'click==7.0'
    ],
    setup_requires=[],
    tests_require=[],
    entry_points="""
        [console_scripts]
        doppel-speller={}.cli:cli
    """.format(module_name)
)
