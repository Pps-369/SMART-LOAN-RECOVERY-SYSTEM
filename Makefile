.PHONY: setup data train score api dashboard test lint docker-up clean

setup:
	pip install -r requirements-dev.txt && pip install -e .

data:
	python -m loan_recovery.data_generator

train:
	python -m loan_recovery.train

score:
	python -m loan_recovery.predict --input data/processed/demo_portfolio.csv --output reports/scored_portfolio.csv

api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run app/dashboard.py

test:
	pytest

lint:
	ruff check src api app tests

docker-up:
	docker compose up --build

clean:
	rm -rf data/raw/*.csv data/processed/*.csv models/*.joblib reports/scored_portfolio.csv
