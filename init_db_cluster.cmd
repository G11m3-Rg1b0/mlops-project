@echo off
:: set environment variables
FOR /F "tokens=* delims=" %%x in (.env) DO set %%x

:: create database cluster
initdb

:: change configuration files to allow connection and modify port
echo host 	all 			all 			0.0.0.0/0				trust >> %PGDATA%/pg_hba.conf
echo port=%POSTGRES_PORT% >> %PGDATA%/postgresql.conf

:: start cluster
pg_ctl -l pgdb_logfile start

:: initialize mlflow database
echo CREATE DATABASE %POSTGRES_DB%; >> init_db.sql
echo CREATE USER %POSTGRES_USER% WITH ENCRYPTED PASSWORD '%POSTGRES_PASSWORD%'; >> init_db.sql
echo GRANT ALL PRIVILEGES ON DATABASE %POSTGRES_DB% TO %POSTGRES_USER%; >> init_db.sql

psql --port=%POSTGRES_PORT% --file=init_db.sql postgres

:: remove sql file
del init_db.sql