@echo off
:: set environment variables
FOR /F "tokens=* delims=" %%x in (.env) DO set %%x

:: start PostgreSQL server
pg_ctl -l logs/pgdb_logfile -D %PGDATA% start

:: wait until ready !infinite loop
echo wait until server is ready ...
:loop
timeout /t 3 /nobreak > NUL
pg_isready -h localhost -p %POSTGRES_PORT% | find "accepting connections" || goto :loop

:: start MLflow server
mlflow server ^
    --backend-store-uri postgresql://%POSTGRES_USER%:%POSTGRES_PASSWORD%@localhost:%POSTGRES_PORT%/%POSTGRES_DB% ^
    --default-artifact-root ./mlruns ^
    || pg_ctl stop