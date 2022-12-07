# MLOpCensus - Machine Learning pipeline for Census data

This project implements a machine learning pipeline to 

- train a model on census data stored in an S3 AWS bucket and version controlled by `DVC`
- deploy the model as a web app using `DVC`, `FastAPI` and `heroku`

To inference the model, use the provided script `starter/post_request.py`. In that file you can enter the census data and do a POST request via (from the root directory and in the virtual environment determined by `requirements.txt`):

```bash
python3 starter/post_request.py
```

The model has been reviewed in `starter/model_card.md`.