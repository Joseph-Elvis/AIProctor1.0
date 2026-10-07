FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /usr/local /usr/local
COPY . /app

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "main_api.py"]
