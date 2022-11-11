from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

from starter.ml.model import inference, get_model

# data object for census model query
class CensusItem(BaseModel):
    # TODO: data types e.g. from sql?
    # TODO: autogenerate this class; at least the categorical values (-> Literals) 
    # to guarantee correctness of data
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int.Field(alias='education-num')
    marital_status: str.Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int.Field(alias='capital-gain')
    capital_loss: int.Field(alias='capital-loss')
    hours_per_week: int.Field(alias='hours-per-week')
    native_country: str.Field(alias='native-country')
    income: Literal[">50K", "<=50K"]

app = FastAPI()

@app.get("/")
async def greeting():
    return {"Welcome to the ML model server!"}

@app.post("/s")
async def model_inf(census_req: CensusItem):
    model = get_model(mode="load")
    # get encoders -> refactor
    # encode CensusItem (ordering in BaseModels?)
    # NOTE: even though we need to convert data the pydantic BaseModel for data checking (no excepts etc.) 
    # return model inference

