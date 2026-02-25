.PHONY: help setup up down logs clean crawl list monitor quota estimate discover add-channel remove-channel history crawl-smart crawl-new dbt-debug dbt-run dbt-test dbt-pipeline prefect-deploy prefect-logs

help:
	@echo "YouTube Analytics Pipeline Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup           - Setup PostgreSQL and BigQuery databases"
	@echo "  make up              - Start Docker services"
	@echo "  make down            - Stop Docker services"
	@echo ""
	@echo "Channel Management:"
	@echo "  make discover        - Search and add channels from CSV"
	@echo "  make add-channel     - Add single channel by name"
	@echo "  make list            - List all channels in database"
	@echo "  make remove-channel  - Remove channel from database"
	@echo ""
	@echo "Data Crawling:"
	@echo "  make crawl-smart     - Smart crawl (auto full/incremental, limit 20)"
	@echo "  make crawl-new       - Crawl never-crawled channels (limit 10)"
	@echo "  make crawl           - Crawl scheduled channels (limit 10)"
	@echo ""
	@echo "Monitoring:"
	@echo "  make quota           - Check API quota status"
	@echo "  make history         - View crawl history"
	@echo "  make monitor         - Full system status check"
	@echo ""
	@echo "DBT:"
	@echo "  make dbt-run         - Run all dbt models"
	@echo "  make dbt-test        - Run all dbt tests"
	@echo "  make dbt-pipeline    - Full pipeline (deps -> run -> test)"
	@echo ""
	@echo "Prefect:"
	@echo "  make prefect-deploy  - Deploy Prefect workflows"
	@echo "  make prefect-logs    - View Prefect worker logs"

setup:
	python -m extract.cli setup

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

discover:
	python -m extract.cli discover config/channels_template.csv --output config/channels_found.csv

add-channel:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make add-channel NAME=\"Channel Name\""; \
		echo "Example: make add-channel NAME=\"@MrBeast\""; \
		exit 1; \
	fi
	python -m extract.cli add-by-name "$(NAME)"

list:
	python -m extract.cli list

remove-channel:
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make remove-channel ID=channel_id"; \
		exit 1; \
	fi
	python -m extract.cli remove $(ID)

crawl-smart:
	python -m extract.cli crawl-smart --limit 20

crawl-new:
	python -m extract.cli crawl-new --limit 10

crawl:
	python -m extract.cli crawl-scheduled --limit 10

quota:
	python -m extract.cli quota

history:
	python -m extract.cli history --limit 20

monitor:
	python script/monitor_quota.py

dbt-debug:
	python script/dbt_cli.py debug

dbt-run:
	python script/dbt_cli.py run

dbt-test:
	python script/dbt_cli.py test

dbt-pipeline:
	python script/dbt_cli.py pipeline

dbt-staging:
	python script/dbt_cli.py run --select staging.*

dbt-mart:
	python script/dbt_cli.py run --select mart.*

prefect-deploy:
	python script/deploy_prefect.py

prefect-logs:
	docker compose logs -f prefect-worker

prefect-restart:
	docker compose restart prefect-worker

clean:
	docker compose down -v
