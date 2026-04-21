.PHONY: install data train eval test serve docker clean

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

train:
	python scripts/train.py --yolo-epochs 5 --yolo-imgsz 256 --yolo-fraction 0.12 --cnn-epochs 1 --kp-limit 1200

eval:
	python scripts/evaluate.py

test:
	pytest -q

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

docker:
	docker build -f docker/Dockerfile -t signlang:latest .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
