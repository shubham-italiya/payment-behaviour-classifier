.PHONY: setup data train test lint all

setup:            ## install the package and dev tools
	pip install -e ".[dev]"

data:             ## download the dataset and build data/payments.db
	python -m payclf.data

train:            ## run the experiment, write reports/
	python -m payclf.train

test:
	pytest -q

lint:
	ruff check .

all: data train test lint
