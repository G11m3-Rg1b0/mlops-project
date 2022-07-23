
# MLOps-project
This project is for me a way to experiment with new tools used in MLOps projects and improve my software development skills.
The initial code of the project is from [Classify MNIST Audio using Spectrograms/Keras CNN](https://www.kaggle.com/code/christianlillelund/classify-mnist-audio-using-spectrograms-keras-cnn/notebook).


## Presentation of the project
In the project I use [DVC](https://dvc.org/) to version control data and models with Git.
I implemented [MLflow](https://mlflow.org/) with [Docker](https://docs.docker.com/) to run as a server for tracking experiments like presented in [scenario 4](https://mlflow.org/docs/latest/tracking.html#scenario-4-mlflow-with-remote-tracking-server-backend-and-artifact-stores)
of MLflow documentation.

<p align="center">
<img src="https://mlflow.org/docs/latest/_images/scenario_4.png" height="220em">
</p>

The backend store is a PostgreSQL database and the model registry is a local file store.
I added an extra layer for security to access the remote server via a reverse proxy with [nginx](https://nginx.org/en/).


## Environment Variables
To run this project, you will need to add the following environment variables in an .env file located in the `docker/` directory
to initialize the database and run the remote server.

`POSTGRES_USER`

`POSTGRES_PASSWORD`

`POSTGRES_DB`

`POSTGRES_PORT`


## Installation
After cloning the repository of this project, create a virtual environment and install the dependencies:
```bash
    pip install -r requirements.txt
```
You will need to have Docker installed and running on your computer.


## Run Locally
To start the remote server in project directory use the command:
```bash
    source server.sh start
```
this will launch the containers that will record the experiments. The server will listen to the port `1024` of the localhost.
Make sure to have your docker service up and running before starting the server.
To stop the server simply modify the argument from `start` to `stop`.

Then you can run the entire DAG with:
```bash
    dvc repro
```


## Running Tests
To run tests, run the following command:
```bash
  pytest tests
```


## Skills Learned
At the current stage of this project I developed several skills:
- Productionize code with [SOLID](https://simple.wikipedia.org/wiki/SOLID_(object-oriented_design)) and [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) principles.
- Data and model versioning with [DVC](https://dvc.org/).
- Model experiment tracking with [MLflow](https://mlflow.org/).
- Containerizing a remote server to safely track experiments with [Docker](https://docs.docker.com/) and [nginx](https://nginx.org/en/).

## Future developments
- Airflow environment for final production step and monitoring.


## Authors
- [@G11m3-Rg1b0](https://www.github.com/G11m3-Rg1b0)


## Feedback
If you have any feedback, please reach out to me at [regimbeauguillaume@gmail.com](mailto:regimbeauguillaume@gmail.com).
