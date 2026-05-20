FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install poetry --break-system-packages && \
    poetry config virtualenvs.create false && \
    poetry install --without dev,ml

COPY . .
RUN pip install -e . --break-system-packages

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]