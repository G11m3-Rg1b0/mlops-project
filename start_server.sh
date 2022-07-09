#! /bin/bash

## start database server
#pg_ctl -l logs/pgdb_log -D ${PGDATA} start
#
## check if database server is ready
#pg_isready -h localhost -p ${POSTGRES_PORT} | grep "accepting connections" || echo "server not ready"
#
## start mlflow server
#mlflow server \
#  --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB} \
#  --default-artifact-root ./mlruns \
#  || pg_ctl -D ${PGDATA} stop

docker-compose --env-file ./docker/.env up

# shut down: docker-compose --env-file ./docker/.env down