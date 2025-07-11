# Use a minimal Python base
FROM python:3.10-slim

WORKDIR /app

# Copy precompiled requirements
COPY requirements.txt .

# Install base deps
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir uvicorn

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
