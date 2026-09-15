FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    LOAN_RECOVERY_ROOT=/app

WORKDIR /app

# install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e . \
    && python -m loan_recovery.train   # model is baked into the image, so containers start ready

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
