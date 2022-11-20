import json
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

    def to_dataframe(self) -> pd.DataFrame:
        # modify dict for converting to dataframe 
        # this avoids the ValueError "... you must pass an index" (alternatively use an index?)
        d = self.dict()
        d = {key: [item] for key, item in d.items()}
        return pd.DataFrame(d)


app = FastAPI()


@app.get("/")
async def greeting():
    return {"Welcome to the ML model server!"}



@app.post("/inference/")
async def model_inf(census_req: CensusItem):
    # check data
    # try:
    #     census_req = CensusItem.parse_obj(census_req)
    # except Exception as exc:
    #     raise exc
    model = get_model(mode="load")
    census_req_df = census_req.to_dataframe()
    pred = inference_pd(model, census_req_df, decode=True)
    return json.dumps(pred.tolist()) # needed for FastAPI responses (jsonable_encoder)
    # get encoders -> refactor
    # encode CensusItem (ordering in BaseModels?)
    # NOTE: even though we need to convert data the pydantic BaseModel for data checking (no excepts etc.)
    # return model inference
