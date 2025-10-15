# MLOps Project

This repository is used to manage a sentiment analysis model using the CI/CD pipeline.

The model is updated and saved on the HuggingFace platform.

You can interact with it using the Python libraries `uvicorn` and `fastapi`.

The prediction results will be saved in a `csv` file.

## Get started

To start using this environment, you need to pull the docker image in your local machine

```
docker pull amosdisabatino/mlops_project
```

...then run it:

```
docker run -p 8000:8000 amosdisabatino/mlops_project
```
