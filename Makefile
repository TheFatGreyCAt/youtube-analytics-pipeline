.PHONY: help setup up down logs clean crawl list dbt-debug dbt-run dbt-test dbt-pipeline dbt-staging dbt-mart monitor quota-check estimate prefect-deploy prefect-logs prefect-restart

help:
	@echo "YouTube Analytics Pipeline Commands"
	@echo "===================================="
	@echo "Setup & Services:"
	@echo "  make setup       - Setup databases"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop services"
	@echo "  make logs        - View logs"
	@echo "  make clean       - Remove all"
	@echo ""
	@echo "Prefect Deployment:"
	@echo "  make prefect-deploy  - Deploy workflows with schedule"
	@echo "  make prefect-logs    - View Prefect worker logs"
	@echo "  make prefect-restart - Restart Prefect worker"
	@echo ""
	@echo "Monitoring & Status:"
	@echo "  make monitor     - Check quota & system status"
	@echo "  make quota-check - Check API quota only"
	@echo "  make estimate N=15 - Estimate cost for N channels"
	@echo ""
	@echo "Data Extraction:"
	@echo "  make crawl       - Crawl channels (limit 10)"
	@echo "  make crawl-15    - Crawl 15 channels"
	@echo "  make list        - List channels"
	@echo ""
	@echo "DBT Commands:"
	@echo "  make dbt-debug    - Check dbt connection"
	@echo "  make dbt-pipeline - Run full dbt pipeline (deps -> run -> test)"
	@echo "  make dbt-run      - Run all dbt models"
	@echo "  make dbt-test     - Run all dbt tests"
	@echo "  make dbt-staging  - Run staging models only"
	@echo "  make dbt-mart     - Run mart models only"

setup:
	python -m extract.cli setup

up:
	docker compose up -d
	@echo "Prefect: http://localhost:4200"
	@echo "Dashboard: http://localhost:8501"

down:
	docker compose down

logs:
	docker compose logs -f

# Prefect deployment
prefect-deploy:
	python script/deploy_prefect.py

prefect-logs:
	docker compose logs -f prefect-worker

prefect-restart:
	docker compose restart prefect-worker

# Monitoring commands
monitor:
	python script/monitor_quota.py

quota-check:
	python script/monitor_quota.py --quota-only

estimate:
	python script/monitor_quota.py --estimate $(N)

# Crawl commands
crawl:
	python -m extract.cli crawl-file --limit 10

crawl-15:
	@echo "⚠️  Crawling 15 channels. Check quota first!"
	python script/monitor_quota.py --quota-only
	@echo ""
	@read -p "Continue? [y/N]: " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python -m extract.cli crawl-file --limit 15; \
	fi

list:
	python -m extract.cli channels

# DBT Commands
dbt-debug:
	python script/dbt_cli.py debug

dbt-pipeline:
	python script/dbt_cli.py pipeline

dbt-run:
	python script/dbt_cli.py run

dbt-test:
	python script/dbt_cli.py test

dbt-staging:
	python script/dbt_cli.py run --select staging.*

dbt-intermediate:
	python script/dbt_cli.py run --select intermediate.*

dbt-mart:
	python script/dbt_cli.py run --select mart.*

dbt-build:
	python script/dbt_cli.py build

dbt-deps:
	python script/dbt_cli.py deps

clean:
	docker compose down -v
