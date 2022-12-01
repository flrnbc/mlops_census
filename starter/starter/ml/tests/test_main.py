import copy
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
        "education-num": 8,
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

DATA2 = {
        "age": 27,
        "workclass": "Private",
        "fnlwgt": 61800,
        "education": "Bachelors",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 10000,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
}


def test_CensusItem():
    """Just a smoke test
    TODO: better assert something?
    """
    ci = CensusItem.parse_obj(DATA)
    print(ci)
    print(ci.dict())
    print(ci.to_dataframe())


def test_api_get():
    r = requests.get("http://127.0.0.1:8000/")
    assert r.status_code == 200
    assert r.json() == "Welcome to the ML model server!"
 

def test_api_post():
    r = requests.post("http://127.0.0.1:8000/inference/", json=DATA)
    assert r.status_code == 200
    assert r.json() == '[">50K"]'


def test_api_post2():
    # TODO: refactor these two tests?!
    r = requests.post("http://127.0.0.1:8000/inference/", json=DATA2)
    assert r.status_code == 200
    assert r.json() == '[">50K"]'


def test_api_post_failure():
    FALSE_DATA = copy.deepcopy(DATA) # deep copy just to be sure (i.e. not side effects on DATA)
    del FALSE_DATA["age"]
    FALSE_DATA["agw"] = 53 # add typo
    r = requests.post("http://127.0.0.1:8000/inference/", json=FALSE_DATA)
    assert r.status_code == 422
    FALSE_DATA["age"] = 67 # add another entry
    r = requests.post("http://127.0.0.1:8000/inference/", json=DATA)
    assert r.status_code == 200 # ok because all necessary data is provided 


