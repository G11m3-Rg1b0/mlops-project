import mlflow
import yaml
import os
import dvc.api

from src.model.model import CNNModel

# todo --> this is messy: constants and variables are mixed

with open(os.path.join('configs', 'base_cfg.yaml'), 'r') as fp_cfg:
    base_config = yaml.safe_load(fp_cfg)

with open('params.yaml', 'r') as fp_p:
    params_config = yaml.safe_load(fp_p)

model_dir = base_config['base']['local_model_registry']
data_train_dir = os.path.join(base_config['base']['dir_preprocessed_data'], 'train')
# path to save info about last mlflow run
path_info = base_config['base']['mlflow_last_run']

experiment_name = params_config['model_training']['experiment']
run_name = params_config['model_training']['run_name']

#####

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


def mlflow_wrapper(func):
    def inner(*args, **kwargs):
        mlflow.set_tracking_uri('http://localhost:5000')

        mlflow.keras.autolog()

        print('start experiment:', experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            exp_id = mlflow.create_experiment(name=experiment_name)
        else:
            exp_id = experiment.experiment_id

        with mlflow.start_run(run_name=run_name, experiment_id=exp_id) as run:
            # Display some information
            artifact_uri_ = mlflow.get_artifact_uri()
            tracking_uri_ = mlflow.get_tracking_uri()
            run_id = run.info.run_id
            print('artifact URI:', artifact_uri_)
            print('tracking URI:', tracking_uri_)
            print(f'run id: {run_id}')
            print(f'exp id: {exp_id}')

            func(*args, **kwargs)

            # add data url to logs
            data_url = dvc.api.get_url(data_train_dir, remote='data-registry')
            mlflow.log_param('data_url', data_url)

            save_run_info(run_id, exp_id)

    return inner
