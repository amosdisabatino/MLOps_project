FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --upgrade pip
RUN pip install -e .

COPY src ./src
COPY tests ./tests

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
