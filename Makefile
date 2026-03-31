.PHONY: generate train-baselines train-cnn evaluate figures v3 all test lint clean

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

v3: figures
	python experiments/run_conformal.py
	python experiments/run_cost_analysis.py
	python experiments/run_coral.py
	python experiments/generate_v3_figures.py

all: generate train-baselines train-cnn evaluate figures v3

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ experiments/

clean:
	rm -rf results/ data/ models/ docs/figures/*.png docs/figures/_generated/
