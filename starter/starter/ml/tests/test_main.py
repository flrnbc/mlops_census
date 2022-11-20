import json

import pydantic
import pytest
import requests
from fastapi.testclient import TestClient
from main import CensusItem, app

client = TestClient(app)


DATA = {
        "age": 53,
        "workclass": "Federal-gov",
        "fnlwgt": 8281,
        "education": "Doctorate",
        "education-num": 891213,
        "marital-status": "Divorced",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 1500,
        "capital-loss": 500,
        "hours-per-week": 30,
        "native-country": "Jamaica"
    }

def test_CensusItem():
    ci = CensusItem.parse_obj(DATA)
    print(ci)
    print(ci.dict())
    print(ci.to_dataframe())


def test_api_post():
   r = requests.post("http://127.0.0.1:8000/inference/", json=DATA)
   print(r.json())


def test_api_post_failure():
    pass