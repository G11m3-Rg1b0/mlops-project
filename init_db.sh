#! /bin/bash

source postgresql.env

# init db
initdb

# change configuration files
echo "host    all             all             0.0.0.0/0               trust" >>"${PGDATA}/pg_hba.conf"
echo "port=${POSTGRES_PORT}" >>"${PGDATA}/postgresql.conf"
UNIX_SOCKET=/tmp
echo "unix_socket_directories = '${UNIX_SOCKET}'" >>"${PGDATA}/postgresql.conf"


# start cluster
pg_ctl -l logs/pgdb_logfile -D ${PGDATA} start

# initialize mlflow database
echo "CREATE DATABASE ${POSTGRES_DB};" >>init_db.sql
echo "CREATE USER ${POSTGRES_USER} WITH ENCRYPTED PASSWORD '${POSTGRES_PASSWORD}';" >>init_db.sql
echo "GRANT ALL PRIVILEGES ON DATABASE ${POSTGRES_DB} TO ${POSTGRES_USER};" >>init_db.sql

psql -host=${UNIX_SOCKET} --port=${POSTGRES_PORT} --file=init_db.sql postgres

# remove sql file
rm init_db.sql
