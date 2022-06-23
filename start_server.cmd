@echo off
:: set environment variables
FOR /F "tokens=* delims=" %%x in (.env) DO set %%x

:: start PostgreSQL server
pg_ctl -l logs/pgdb_logfile start

:: wait until ready !infinite loop
echo wait until server is ready ...
:loop
timeout /t 3 /nobreak > NUL
pg_isready -U %POSTGRES_USER% -d %POSTGRES_DB% | find "accepting connections" || goto :loop

:: start MLflow server
mlflow server ^
    --backend-store-uri postgresql://%POSTGRES_USER%:%POSTGRES_PASSWORD%@localhost:%POSTGRES_PORT%/%POSTGRES_DB% ^
    --default-artifact-root ./mlruns ^
    || pg_ctl stop