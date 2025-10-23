# MLOps Project

This repository is used to manage a sentiment analysis model using the CI/CD pipeline.

The model is updated and saved on the HuggingFace platform.

You can interact with it using the Python libraries `uvicorn` and `fastapi`.

The prediction results will be saved in a `csv` file.

With the interaction of the model, it is possible to classify each sentence in `negative`, `neutral` or `positive`.

## Motivation and Architectural Choices

Transformer-based model: chosen for its excellent NLP capabilities and context management.

FastAPI + Docker: ensure lightweight, scalable and easily containerisable deployment.

Hugging Face Hub: for automatic versioning and model sharing.

CSV Logging: each prediction is saved for analysis and monitoring of real data.

## Fine Tuning Process

The model is re-trained with sentiment data of the `tweet_eval` dataset, in order to classify 3 classes (`negative`, `neutral` and `positive`).
The model trained is `cardiffnlp/twitter-roberta-base-sentiment-latest` in `HuggingFace` and its `tokenizer` is used to prepare the data.

## Use cases:

The model can be used to monitor a company's reputation or the quality of a product for example.

## Get started

To start using this environment, you need to pull the docker image in your local machine

```
docker pull amosdisabatino/mlops_project
```

...then run it:

```
docker run -p 8000:8000 amosdisabatino/mlops_project
```

## Technologies used:

- Docker

- HuggingFace

- Python

- GitHub Actions

- FastAPI

- Pandas
