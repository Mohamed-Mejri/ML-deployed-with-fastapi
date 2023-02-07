from typing import Optional, Union

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.train.constants import CAT_FEATURES
from src.train.ml.model import inference, load_model
from src.train.ml.data import process_data


app = FastAPI()

def to_hyphen(string: str) -> str:
    return string.replace("_", "-")

class Data(BaseModel):
    age: Optional[Union[int, list]] = 30
    workclass: Optional[Union[str, list]] = 'Self-emp-inc'
    fnlgt: Optional[Union[int, list]] = 77516
    education: Optional[Union[str, list]] = 'Bachelors'
    education_num: Optional[Union[int, list]] = 10
    marital_status: Optional[Union[str, list]] = 'Never-married'
    occupation: Optional[Union[str, list]] = 'Exec-managerial'
    relationship: Optional[Union[str, list]] = 'Not-in-family'
    race: Optional[Union[str, list]] = 'Black'
    sex: Optional[Union[str, list]] = 'Female'
    capital_gain: Optional[Union[int, list]] = 6084
    capital_loss: Optional[Union[int, list]] = 0
    hours_per_week: Optional[Union[int, list]] = 40
    native_country: Optional[Union[str, list]] = 'Italy'

    class Config:
        alias_generator = to_hyphen


@app.on_event("startup")
async def startup_event(): 
    global model, encoder, lb
    model = load_model("./src/model/model.pkl")
    encoder = load_model("./src/model/encoder.pkl")
    lb = load_model("./src/model/lb.pkl")

@app.get("/")
def home():
    msg = "Welcome to the 3rd project of the ML Devops Engineer nanodegree developed by Mohamed Mejri"
    return {
        "greetings": msg
    }

@app.post("/api/")
def api(data: Data):
    dic = data.dict(by_alias=True)
    print(dic)
    df = pd.DataFrame([dic])
    print(df)
    
    X, _, _, _ = process_data(
        df, categorical_features=CAT_FEATURES, label=None, training=False,
        encoder=encoder, lb=lb

    )
    preds = inference(model, X)
    print(preds)
    output = lb.inverse_transform(preds)
    print(output)
    return {
        "predictions": str(output)
    }    



