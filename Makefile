.PHONY: generate train-baselines train-cnn evaluate figures all test lint clean

generate:
	python -m simtodata.data.generate

train-baselines:
	python experiments/run_baselines.py

train-cnn:
	python experiments/run_classification.py

evaluate:
	python experiments/run_robustness.py
	python experiments/run_adaptation_curve.py

figures:
	python experiments/generate_figures.py

all: generate train-baselines train-cnn evaluate figures

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ experiments/

clean:
	rm -rf results/ data/ models/ docs/figures/*.png
