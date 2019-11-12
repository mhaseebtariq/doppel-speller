"""A file to define the CLI used to control the recommender-system."""

import json
import logging
from datetime import datetime, timedelta

import click
from datascience.common.tools import attempt_to_remove_file
from datascience.google_cloud.storage import create_environment_specific_bucket_name, download_file_from_gcs
from datascience.google_cloud.big_query import create_standard_sql_views_from_a_gcs_bucket
from datascience.google_cloud.data_proc_executor import DEFAULT_ZONES
from datascience.google_cloud.common import try_to_get_default_gcp_project
from datascience.common import constants as mc
from datascience.monitoring.logger import get_logger

from recommendersystems import __version__, __build__
from recommendersystems.commands.monitoring import operational
from recommendersystems import settings as cs
from recommendersystems.commands.upload.upload_code import code_and_dependencies_upload
from recommendersystems.commands.features import constants as fc
from recommendersystems.commands.features import settings as fs
from recommendersystems.commands.recommendations import settings as rs
from recommendersystems.commands.upload import settings as us

from recommendersystems import cli_utils


LOGGER = get_logger()

CLI_CONFIG = {
    'features': {
        'init_code': us.INITIALISATION_CODE_FOR_FEATURES_GENERATION,
        'job_name': us.JOB_NAME_JOB_GENERATE_FEATURES,
        'accelerators': [(list(), DEFAULT_ZONES)],
        'settings': fs.ENCODING_OUTPUT_STORAGE_SETTINGS_MAPPING
    },
    'training': {
        'init_code': us.INITIALISATION_CODE_FOR_TRAINING_AND_PREDICTION,
        'job_name': us.JOB_NAME_JOB_TRAIN_MODEL,
        'accelerators': rs.DATAPROC_GPU_ACCELERATORS_TO_TRY,
        'settings': rs.TRAINING_STORAGE_SETTINGS_MAPPING
    },
    'recommendations': {
        'init_code': us.INITIALISATION_CODE_FOR_TRAINING_AND_PREDICTION,
        'job_name': us.JOB_NAME_JOB_GENERATE_RECOMMENDATIONS,
        'accelerators': rs.DATAPROC_GPU_ACCELERATORS_TO_TRY,
        'settings': rs.PREDICTION_STORAGE_SETTINGS_MAPPING
    }
}


project_option = click.option(
    '--project',
    help='GCP Project. Default is None - the Python libraries will pick the default in that case!',
    default=try_to_get_default_gcp_project()
)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__)
@click.option('-v', '--verbose', count=True, envvar='LOGGING_LEVEL',
              help='Make output more verbose. Use more v\'s for more verbose')
@click.pass_context
def cli(ctx, verbose):
    print('Recommender Systems v{}-{}'.format(__version__, __build__))

    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    logging.getLogger().setLevel(level)


@cli.command()
@project_option
def upload_code(**kwargs):
    """
    Upload the code to the Google Cloud Storage bucket
    """
    code_and_dependencies_upload({'project': kwargs['project']})


@cli.command()
def dataproc_cluster_monitor():
    """
    Send number of active cluster on dataproc to datadog.
    """
    from recommendersystems.dataproc_cluster_monitoring.main import main
    main()


@cli.command()
@project_option
def create_views_on_bigquery(**kwargs):
    """
    Creates views on BigQuery (for settings.DATA_BIDDING_MANAGEMENT_DATASET)
     - Uploaded to gs://<gcs_bucket>/<gcs_bucket_folder>/
    """
    dataset = cs.BIGQUERY_DATASET_NAME
    project = kwargs['project']
    # This bucket is accessible from all environments
    gcs_bucket = 'coolblue-ds-big-query-views'
    gcs_bucket_folder = cs.BIGQUERY_DATASET_NAME
    create_standard_sql_views_from_a_gcs_bucket(
        dataset=dataset,
        project=project,
        gcs_bucket=gcs_bucket,
        gcs_bucket_folder=gcs_bucket_folder
    )


@cli.command()
def write_latest_sunday_value_to_azkaban_job_output_property_file(**kwargs):
    """
    Work around for passing environment variables to Azkaban job
    """
    latest_sunday = cli_utils.get_latest_sunday()
    cmd = '''/usr/bin/printf '{{"LASTSUNDAY": "{}"}}\\n' > "${{JOB_OUTPUT_PROP_FILE}}"'''.format(latest_sunday)

    with open('write_latest_sunday_value.sh', 'w') as f:
        f.write(cmd)


@cli.command()
@project_option
@click.option('--run_as_of_date',
              help='The day you want to run the cli for. Default is yesterday!',
              default=datetime.today() - timedelta(days=1),
              callback=cli_utils.process_date_param)
@click.option('--last_training_date',
              help='For which date to use the "training" features attributes data. Default is the last Sunday - '
                   'Based on the run_as_of_date. The default is set in the cli (when None)!',
              default=None,
              callback=cli_utils.process_date_param)
@click.option('--training/--prediction',
              help='Whether to create features for "training" or "prediction" dataset - default is True!',
              default=True)
@click.option('--skip_if_model_is_going_to_be_trained_today',
              help='The model is trained (recommendations are generated right after that as well) when - '
                   'datetime.now().isoweekday() == cs.MODEL_TRAINED_ON_ISO_WEEKDAY!',
              is_flag=True,
              default=False)
@click.option('--storage_settings',
              help='The storage settings to use for the run - see ENCODING_OUTPUT_STORAGE_SETTINGS_MAPPING '
                   '(in features/settings.py)!',
              default=fc.STORAGE_LOCAL_SSD_SETTINGS_KEY)
def create_training_or_prediction_features(**kwargs):
    """Create train and prediciton features.

    Arguments:
        project (str): Google Cloud Platform project.
        run_as_of_date (datetime.date): Run start date.
        last_training_date (datetime.date): Training end date.
        training (bool): Create training or prediction data.
        skip_if_model_is_going_to_be_trained_today (bool):
            Flag if model can be skipped.
        storage_settings (dict): Storage settings, see features/settings.py
    """
    if kwargs['training']:
        _create_features(
            cluster_name=fs.DATAPROC_CLUSTER_NAME_TRAINING_FEATURES,
            **kwargs
        )
    else:
        _create_features(
            cluster_name=fs.DATAPROC_CLUSTER_NAME_PREDICTION_FEATURES,
            **kwargs
        )


def _create_features(**kwargs):
    """Create features and run jobs."""
    can_skip_cli = kwargs.pop(
        'skip_if_model_is_going_to_be_trained_today',
        False
    )
    cli_utils.model_training_day(
        cli_name='create_training_or_prediction_features',
        can_skip_cli=can_skip_cli
    )
    cli_utils.monitor_feature_creation(training=kwargs['training'])
    _base_job_run(stage_key='features', **kwargs)


@cli.command()
@project_option
@click.option('--run_as_of_date',
              help='The day you want to run the cli for. Default is the last Sunday!',
              default=cli_utils.get_latest_sunday(),
              callback=cli_utils.process_date_param)
@click.option('--storage_settings',
              help='The storage settings to use for the run - see TRAINING_STORAGE_SETTINGS_MAPPING '
                   '(in recommendations/settings.py)!',
              default=fc.STORAGE_LOCAL_SSD_SETTINGS_KEY)
@click.option('--use_gpu',
              help='Whether to use GPU for training!',
              is_flag=True,
              default=False)
@click.option('--dry_run',
              help='Used to check the execution of the process, it trains the model on a small subset of data.',
              is_flag=True,
              default=False)
def train_recommendations_model(**kwargs):
    """Triggering job to train recommendation model.

    Arguments:
        project (str): Google Cloud Platform project.
        run_as_of_date (datetime.date): Run start date.
        storage_settings (dict): Storage settings, see features/settings.py
        use_gpu (bool): Flag to use GPUs for training.
        dry_run (bool): Flag to run with a small subset of data.
    """
    if (not cli_utils.model_training_day()) and (not kwargs['dry_run']):
        raise Exception('"train_recommendations_model" is only supposed to run when - '
                        'datetime.now().isoweekday() == cs.MODEL_TRAINED_ON_ISO_WEEKDAY')
    cli_utils.monitor_train_recommendations()
    _base_job_run(
        stage_key='training',
        cluster_name=rs.DATAPROC_CLUSTER_NAME_MODEL_TRAINING,
        **kwargs
    )


@cli.command()
@project_option
@click.option('--run_as_of_date',
              help='The day you want to run the cli for. Default is yesterday!',
              default=datetime.today() - timedelta(days=1),
              callback=cli_utils.process_date_param)
@click.option('--storage_settings',
              help='The storage settings to use for the run - see TRAINING_STORAGE_SETTINGS_MAPPING '
                   '(in recommendations/settings.py)!',
              default=fc.STORAGE_LOCAL_SSD_SETTINGS_KEY)
@click.option('--last_training_date',
              help='For which date to use the "trained" model / features attributes data. Default is the last Sunday - '
                   'Based on the run_as_of_date. The default is set in the cli (when None)!',
              default=None,
              callback=cli_utils.process_date_param)
@click.option('--use_gpu',
              help='Whether to use GPU for predictions!',
              is_flag=True,
              default=False)
@click.option('--include_integration_test',
              help='It includes an integration test that predicts on the test set observations and compares against '
                   'the predictions executed at training time',
              is_flag=True,
              default=False)
@click.option('--skip_if_model_is_going_to_be_trained_today',
              help='The model is trained (recommendations are generated right after that as well) when - '
                   'datetime.now().isoweekday() == cs.MODEL_TRAINED_ON_ISO_WEEKDAY!',
              is_flag=True,
              default=False)
def generate_recommendations(**kwargs):
    """
    Loads the latest saved trained model and generates predictions.

    Arguments:
        project (str): Google Cloud Platform project.
        run_as_of_date (datetime.date): Run start date.
        last_training_date (datetime.date): Training end date.
        use_gpu (bool): Flag to use GPUs for training.
        include_integration_test (bool):
            Include integration tests to compare training vs predict values.
        skip_if_model_is_going_to_be_trained_today (bool):
            Flag if model can be skipped.
    """
    can_skip_cli = kwargs.pop('skip_if_model_is_going_to_be_trained_today', False)
    cli_utils.model_training_day(
        cli_name='generate_recommendations',
        can_skip_cli=can_skip_cli
    )
    cli_utils.monitor_generate_recommendations()
    _base_job_run(
        stage_key='recommendations',
        cluster_name=rs.DATAPROC_CLUSTER_NAME_MODEL_PREDICTION,
        **kwargs
    )
    project = kwargs['project']
    run_as_of_date = kwargs['run_as_of_date']
    last_training_date = cli_utils.process_run_dates(
        project=project,
        run_as_of_date=run_as_of_date,
        last_training_date=kwargs['last_training_date']
    )
    cli_utils.base_monitor(
        msg=mc.GENERIC_COUNT_WITH_TAGS_METRIC_DD,
        extra={
            'metric': 'number_of_days_since_the_last_training',
            'count': (run_as_of_date - last_training_date).days,
            'tags': {'date': str(run_as_of_date)}
        }
    )

    # Sending output stats to DataDog
    operational.send_output_information_for_the_run_to_datadog(
        project=kwargs['project'],
        run_date=run_as_of_date,
        logger=LOGGER
    )


@cli.command()
@project_option
def sample_job_for_airflow_delete_later(**kwargs):
    project = kwargs['project']

    process_name = '{}.metric.from.airflow'.format(cli_utils.module_name)
    LOGGER.info('TRYING TO SEND METRIC TO DATADOG')
    LOGGER.monitor(mc.PROCESS_STARTED_DD, process=process_name)

    LOGGER.info('TRYING TO READ FILE FROM GCS')
    gcs_bucket = 'coolblue-ds-tensorflow-gpu-build-files'
    file_location = download_file_from_gcs(project, gcs_bucket, 'sample.txt')
    with open(file_location, 'r') as fl:
        content = fl.read().strip()
        LOGGER.info('FILE CONTENT: {}'.format(content))
    attempt_to_remove_file(file_location)

    LOGGER.monitor(mc.PROCESS_FINISHED_DD, process=process_name)


def _base_job_run(
        stage_key,
        cluster_name,
        project,
        storage_settings,
        run_as_of_date,
        last_training_date=None,
        training=None,
        use_gpu=None,
        dry_run=None,
        include_integration_test=None
):
    """Base job called by the feature, training, and recommendation stages.

    Arguments:
        stage_key (str): The stage calling the base_job. Only 'features',
            'training', and 'recommendations' are available keys.
        cluster_name (str): Name of cluster
        project(str): Name of project
        storage_settings (dict): Storage settings, see features/settings.py
        run_as_of_date (datetime.date): Run start date.
        last_training_date (datetime.date): Training end date.
        training (bool): Create training or prediction data.
        use_gpu (bool): Flag to use GPUs for training.
        dry_run (bool): Flag to run with a small subset of data.
        include_integration_test (bool):
            Include integration tests to compare training vs predict values.
    """
    config = CLI_CONFIG[stage_key]
    args = _dataproc_args(
        stage_key=stage_key,
        project=project,
        run_as_of_date=run_as_of_date,
        last_training_date=last_training_date,
        storage_settings=storage_settings,
        training=training,
        use_gpu=use_gpu,
        dry_run=dry_run,
        include_integration_test=include_integration_test
    )
    gcs_bucket = create_environment_specific_bucket_name(
        base_name=cs.GCS_BUCKET_NAME,
        project=project
    )

    init_actions = cli_utils.init_actions(
        gcs_bucket=gcs_bucket,
        init_code=config['init_code']
    )
    job_path = cli_utils.job_path(
        gcs_bucket=gcs_bucket,
        job_name=config['job_name']
    )

    dataproc = None
    for accelerators, zones_to_try in config['accelerators']:
        dataproc = cli_utils.create_single_node_dataproc_cluster_given_settings(
            settings_key=storage_settings,
            settings=config['settings'][storage_settings],
            project=project,
            cluster_name=cluster_name,
            init_actions=init_actions,
            accelerators=accelerators,
            zones_to_try=zones_to_try
        )
        if dataproc.zone is not None:
            break

    if stage_key in ('training', 'recommendations'):
        cli_utils.check_if_tensorflow_compiled_for_the_chosen_dataproc_zone(
            dataproc=dataproc,
            raise_exception=False
        )

    dataproc.execute_dataproc_job(
        script=job_path,
        python_files=cli_utils.python_files_locations(gcs_bucket=gcs_bucket),
        args=args
    )
    dataproc.delete_dataproc_cluster()
    LOGGER.monitor(mc.PROCESS_FINISHED_DD)


def _dataproc_args(
        stage_key,
        project,
        run_as_of_date,
        last_training_date,
        storage_settings,
        training,
        use_gpu,
        dry_run,
        include_integration_test
):
    config = CLI_CONFIG[stage_key]
    last_training_date = cli_utils.process_run_dates(
        project=project,
        run_as_of_date=run_as_of_date,
        last_training_date=last_training_date
    )

    args = {
        'project': project,
        'run_as_of_date': str(run_as_of_date),
        'storage_root_folder': (
            (
                config['settings'][storage_settings]
                [fc.STORAGE_ROOT_DIRECTORY_KEY]
            )
        ),
    }
    if stage_key == 'features':
        args['last_training_date'] = str(last_training_date)
        args['run_training_job'] = training
    elif stage_key == 'training':
        args['use_gpu'] = use_gpu
        args['dry_run'] = dry_run
    elif stage_key == 'recommendations':
        args['last_training_date'] = str(last_training_date)
        args['use_gpu'] = use_gpu
        args['include_integration_test'] = include_integration_test
    return json.dumps(args)


if __name__ == '__main__':
    cli()
