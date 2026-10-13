.PHONY: setup synthetic-data test inference model pipeline evaluate serve clean

setup:
	pip install -r requirements.txt

synthetic-data:
	python src/generate_synthetic.py

test:
	pytest tests/ -v

inference:
	python src/train_logistic.py

model:
	python src/train_gbm.py

pipeline:
	python src/pipeline.py

evaluate:
	python src/evaluate.py

serve:
	uvicorn api:app --reload --app-dir src

clean:
	rm -rf data/raw/*.csv data/processed/*.csv models/*.txt reports/*.json reports/*.png
