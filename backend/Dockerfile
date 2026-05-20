FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY crip_x/ ./crip_x/
COPY api/ ./api/
COPY data/ ./data/

RUN pip install poetry --break-system-packages && \
    poetry config virtualenvs.create false && \
    poetry install --without dev,ml --no-root && \
    pip install pydantic-settings --break-system-packages

RUN pip install -e . --break-system-packages

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]