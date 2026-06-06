FROM python:3.10-slim

ARG REQUIREMENTS_FILE=requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/raw/hf_cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements*.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r "${REQUIREMENTS_FILE}"

COPY src ./src
COPY configs ./configs
COPY docs ./docs
COPY notebooks ./notebooks

RUN python -m pip install -e .

EXPOSE 8501

CMD ["python", "-m", "similarity_search.data.eda_allnli", "--help"]

