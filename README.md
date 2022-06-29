
# MLOps-project

This project is for me a way to experiment with new tools required in MLOps projects and improve my software development skills.
This means that the main purpose of the project is not to be user friendly for now, but it will be a future goal for sure.
I like to challenge myself with not so straight forward examples and the initial code of the project is from [Classify MNIST Audio using Spectrograms/Keras CNN](https://www.kaggle.com/code/christianlillelund/classify-mnist-audio-using-spectrograms-keras-cnn/notebook).

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file to initialize the database and run the server with MLflow

`PGDATA`

`POSTGRES_USER`

`POSTGRES_PASSWORD`

`POSTGRES_DB`

`POSTGRES_PORT`

## Installation

After cloning the repository of this project, create a virtual environment and install the dependencies:
```bash
    pip install -r requirements.txt
```

Then create the PostgreSQL database for MLflow backend store in bash command and finalize MLflow setup:
```bash
    init_db_cluster
```

## Run Locally

This part might be tricky to do since I kept some local folder as 'dummy remote' for me to experiment, especially for the DVC setup. However, if you managed the installation process you can directly start the MLflow server delivered at https://localhost:5000 with the command:
```bash
    start_server
```

and then run the DAG with:
```bash
    dvc repro
```
to experiment some model.

## Running Tests

To run tests, run the following command:

```bash
  pytest tests
```


## Lessons Learned

At the current stage of this project I developped several skills:
- Productionize code with [SOLID](https://simple.wikipedia.org/wiki/SOLID_(object-oriented_design)) and [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) principles.
- Data and model versioning with [DVC](https://dvc.org/).
- Model experiment tracking with [MLflow](https://mlflow.org/).

## Future developments

- Docker setup for easy deployment.
- Airflow environment for production and monitoring.


## Authors

- [@G11m3-Rg1b0](https://www.github.com/G11m3-Rg1b0)


## Feedback

If you have any feedback, please reach out to me at [regimbeauguillaume@gmail.com](mailto:regimbeauguillaume@gmail.com).

