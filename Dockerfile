FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY web ./web
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["python", "-m", "ecg_ai_monitor.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
