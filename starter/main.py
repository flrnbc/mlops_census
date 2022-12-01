"""Main module to call model inference via FastAPI"""
import json
import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.model import get_model, inference, inference_pd

# from typing import Literal


# data object for census model query
class CensusItem(BaseModel):
    # TODO: data types e.g. from sql?
    # TODO: autogenerate this class; at least the categorical values (-> Literals)
    # to guarantee correctness of data
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        """Provide example input"""
        schema_extra = {
            "example": {
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
                "native-country": "Jamaica",
            }
        }

    def to_dataframe(self) -> pd.DataFrame:
        # modify dict for converting to dataframe 
        # this avoids the ValueError "... you must pass an index" (alternatively use an index?)
        d = self.dict()
        d = {key: [item] for key, item in d.items()}
        return pd.DataFrame(d)


app = FastAPI()


@app.get("/")
async def greeting():
    return "Welcome to the ML model server!"


@app.post("/inference/")
async def model_inf(census_req: CensusItem):
    """Call model inference"""
    model = get_model(mode="load")
    census_req_df = census_req.to_dataframe()
    pred = inference_pd(model, census_req_df, decode=True)
    return json.dumps(pred.tolist()) # needed for FastAPI responses (jsonable_encoder)


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")