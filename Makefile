.PHONY: install index api ui demo eval test docker-up docker-down

install:
	pip install -e ".[dev]"
	python -m spacy download en_core_web_sm

index:
	python ingestion/build_index.py

api:
	uvicorn api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py --server.port 8501

demo:
	python demo.py --offline

demo-api:
	python demo.py

eval:
	python eval/run_ragas.py

test:
	pytest tests/ -v

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api
