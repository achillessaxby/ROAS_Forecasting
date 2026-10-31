# One image, three roles (see docker-compose.yml): pipeline job, Forecast API, Analytics Suite.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/srv/src:/srv
# libgomp1 is required by XGBoost; curl is used by the container health checks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY pyproject.toml README.md requirements-lock.txt ./
RUN pip install -r requirements-lock.txt   # pinned, Docker-verified versions
COPY src ./src
COPY services ./services
COPY run_demo.py ./
# make the code readable by the non-root user regardless of the host checkout's file permissions
RUN chmod -R a+rX /srv && pip install --no-deps ".[api,app,xgboost]"

# run as an unprivileged user; /data and /registry are mounted volumes
RUN useradd --create-home --uid 1000 roas && mkdir -p /data /registry && chown roas:roas /data /registry
USER roas
EXPOSE 8080 8501
CMD ["python", "run_demo.py", "--data", "/data/synthetic", "--registry", "/registry"]
