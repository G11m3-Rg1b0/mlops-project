FROM python:3.7-slim-buster

ARG MLFLOW_VERSION=1.27.0

RUN pip install mlflow==$MLFLOW_VERSION \
    && pip install psycopg2-binary

# add packages to run training script
