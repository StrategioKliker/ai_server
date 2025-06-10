# Get python 
FROM python:3.12-slim 

# GET UV from their base image 
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

# Set workdirectory 
WORKDIR /app 

# Install python dependencies
COPY pyproject.toml uv.lock ./

RUN uv sync --locked

# Copy source code 
COPY . . 

# Expose port 
EXPOSE 8000 

# Activate venv by putting it in PATH
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]