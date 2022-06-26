import mlflow
import yaml
import os
import dvc.api

from src.model.model import CNNModel
from mlflow.tracking.fluent import ActiveRun

# todo --> this is messy: constants and variables are mixed: write constant in a config.py file ?

with open(os.path.join('configs', 'base_cfg.yaml'), 'r') as fp_cfg:
    base_config = yaml.safe_load(fp_cfg)

with open('params.yaml', 'r') as fp_p:
    params_config = yaml.safe_load(fp_p)

# CONSTANTS
model_dir = base_config['base']['local_model_registry']
data_train_dir = os.path.join(base_config['base']['dir_preprocessed_data'], 'train')
# path to save info about last mlflow run
path_info = base_config['base']['mlflow_last_run']

# variables
experiment_name = params_config['model_training']['experiment']
run_name = params_config['model_training']['run_name']


#####

# todo -> group those
def save_run_info(run_id: str, exp_id: str) -> None:
    """Update 'mlflow_last_run' configuration file with experiment and training run id for upcoming evaluations."""
    info = {
        'run_id':        run_id,
        'experiment_id': exp_id
    }
    with open(path_info, 'w') as fp:
        yaml.safe_dump(info, fp)


def load_model() -> CNNModel:
    """Load the model from the default mlflow model registry.

    return:
        The mlflow saved model.
    """
    with open(path_info, 'r') as fp:
        last_run = yaml.safe_load(fp)

    path_to_model = os.path.join(model_dir, last_run['experiment_id'], last_run['run_id'], 'artifacts', 'model')
    return mlflow.keras.load_model(path_to_model)


######################################


##### params to log
logged_params = {
    # log the current url of the data used for training
    'data_url': dvc.api.get_url(data_train_dir, remote='data-registry'),
}


##############################

# class SetupMLflow

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


def _add_new_logs(params) -> None:
    """Add new logs to the current active run.

    return:
        None.
    """
    mlflow.log_params(params)


def _setup_mlflow() -> str:
    """Setup tracking uri, keras autologging and experiment id for the upcoming runs.

    return:
        Experiment id for the upcoming run.
    """
    mlflow.set_tracking_uri('http://localhost:5000')
    tracking_uri_ = mlflow.get_tracking_uri()
    print('tracking URI:', tracking_uri_)

    mlflow.keras.autolog()

    print('start experiment:', experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        exp_id = mlflow.create_experiment(name=experiment_name)
    else:
        exp_id = experiment.experiment_id
    return exp_id


def mlflow_wrapper(func):
    def inner(*args, **kwargs):
        exp_id = _setup_mlflow()

        with mlflow.start_run(run_name=run_name, experiment_id=exp_id) as run:
            _display_info(run)

            func(*args, **kwargs)

            _add_new_logs(logged_params)

            save_run_info(run.info.run_id, exp_id)

    return inner
