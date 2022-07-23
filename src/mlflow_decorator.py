import mlflow
import os
import dvc.api

from mlflow.tracking.fluent import ActiveRun
from src.utils import (
    save_run_info,
    get_base_config,
    get_params_config,
    get_last_run_info
)

## debugging
# from mlflow.utils.autologging_utils import _logger
# _logger.setLevel('DEBUG')


# todo: should be in the init file (constants and maybe params too)cd
# CONSTANTS
base_config = get_base_config()
data_train_dir = os.path.join(base_config['dir_preprocessed_data'], 'train')

# PARAMS
params_config = get_params_config()
experiment_name = params_config['model_training']['experiment']
run_name = params_config['model_training']['run_name']

train_params = {
    # log the current url of the data used for training
    'data_url': dvc.api.get_url(data_train_dir, remote='data-registry'),
}


def mlflow_run(func):
    def inner(*args, **kwargs):
        exp_id = _setup_mlflow()

        with mlflow.start_run(run_name=run_name, experiment_id=exp_id) as run:
            _display_info(run)

            func(*args, **kwargs)

            _add_logs(train_params)

            save_run_info(run.info.run_id, exp_id, run.info.artifact_uri)

    return inner


def _display_info(run: ActiveRun) -> None:
    """Display some information about the current active run.

    return:
        None.
    """
    artifact_uri_ = run.info.artifact_uri
    run_id = run.info.run_id
    exp_id = run.info.experiment_id

    print('artifact URI:', artifact_uri_)
    print(f'run id: {run_id}')
    print(f'exp id: {exp_id}')


def _add_logs(params: dict) -> None:
    """Add several items to the current active run's logs.

    return:
        None.
    """
    mlflow.log_params(params)


def _setup_mlflow() -> str:
    """Setup tracking uri, keras autologging and experiment id for the upcoming runs.

    return:
        Experiment id for the upcoming run.
    """
    # setting tracking uri
    mlflow.set_tracking_uri('http://localhost:1024')

    # setting autologging for the run
    mlflow.keras.autolog()

    # setting experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        exp_id = mlflow.create_experiment(name=experiment_name)
    else:
        exp_id = experiment.experiment_id

    tracking_uri = mlflow.get_tracking_uri()
    print('tracking URI:', tracking_uri)
    print('start experiment:', experiment_name)

    return exp_id
