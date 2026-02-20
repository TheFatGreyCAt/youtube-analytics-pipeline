.PHONY: help setup up down logs clean crawl list

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
