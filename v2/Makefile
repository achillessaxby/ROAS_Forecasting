# Convenience targets. Everything here also works without make: see README.md.
AS_OF ?= 2025-09-01

# --- Docker (recommended) ---------------------------------------------------
up:        ## build and start pipeline -> api + app (Ctrl-C to stop)
	@test -f .env || cp .env.example .env
	docker compose up --build

up-d:      ## same, in the background
	@test -f .env || cp .env.example .env
	docker compose up --build -d

smoke:     ## check the running API (auth, forecast, validation)
	python scripts/smoke_test.py

down:      ## stop and remove containers (keeps ./data and ./registry)
	docker compose down

clean:     ## remove generated data and models
	rm -rf data registry

# --- Plain Python -----------------------------------------------------------
install:
	pip install -e ".[api,app,xgboost,dev]"

demo:
	python run_demo.py --as-of $(AS_OF)

serve:
	ROAS_REGISTRY=registry ROAS_API_KEYS=local-demo-key uvicorn services.api.main:app --port 8080

app:
	ROAS_REGISTRY=registry ROAS_REPORTS=registry/reports streamlit run services/app/Home.py

test:
	pytest

.PHONY: up up-d smoke down clean install demo serve app test
