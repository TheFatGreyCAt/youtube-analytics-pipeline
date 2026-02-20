.PHONY: help setup up down logs clean crawl list dbt-debug dbt-run dbt-test dbt-pipeline dbt-staging dbt-mart

help:
	@echo "YouTube Analytics Pipeline Commands"
	@echo "===================================="
	@echo "make setup    - Setup databases"
	@echo "make up       - Start all services"
	@echo "make down     - Stop services"
	@echo "make logs     - View logs"
	@echo "make crawl    - Run crawl"
	@echo "make list     - List channels"
	@echo "make clean    - Remove all"
	@echo ""
	@echo "DBT Commands"
	@echo "============"
	@echo "make dbt-debug    - Check dbt connection"
	@echo "make dbt-pipeline - Run full dbt pipeline (deps -> run -> test)"
	@echo "make dbt-run      - Run all dbt models"
	@echo "make dbt-test     - Run all dbt tests"
	@echo "make dbt-staging  - Run staging models only"
	@echo "make dbt-mart     - Run mart models only"

setup:
	python -m extract.cli setup

up:
	docker-compose up -d
	@echo "Prefect: http://localhost:4200"
	@echo "Dashboard: http://localhost:8501"

down:
	docker-compose down

logs:
	docker-compose logs -f

crawl:
	python -m extract.cli crawl --limit 10

list:
	python -m extract.cli list

clean:
	docker-compose down -v

# DBT Commands
dbt-debug:
	python dbt_cli.py debug

dbt-pipeline:
	python dbt_cli.py pipeline

dbt-run:
	python dbt_cli.py run

dbt-test:
	python dbt_cli.py test

dbt-staging:
	python dbt_cli.py run --select staging.*

dbt-intermediate:
	python dbt_cli.py run --select intermediate.*

dbt-mart:
	python dbt_cli.py run --select mart.*

dbt-build:
	python dbt_cli.py build

dbt-deps:
	python dbt_cli.py deps
