.PHONY: setup test demo yahoo bloomberg clean

setup:
	python -m venv .venv && .venv/bin/pip install -r requirements.txt

test:
	python -m tests.test_metrics

demo:
	python tools/make_demo_data.py
	python -m src.run --source demo

yahoo:
	python -m src.run --source yahoo

bloomberg:
	python -m src.run --source csv

# force a fresh pull rather than using the cached prices
refresh:
	python -m src.run --source yahoo --no-cache

clean:
	rm -f output/*.pdf output/*.xlsx output/*.md data/raw/*.csv
	rm -rf src/__pycache__ tests/__pycache__ config/_universe_subset.csv
