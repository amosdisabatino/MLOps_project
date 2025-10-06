FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY tests ./tests

RUN pip install --upgrade pip
RUN pip install .
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')"

CMD ["python", "-m", "src.model"]
