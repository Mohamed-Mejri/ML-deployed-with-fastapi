from typing import Optional, Union

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.train.constants import CAT_FEATURES
from src.train.ml.model import inference, load_model
from src.train.ml.data import process_data


app = FastAPI()

class Data(BaseModel):
    age: Optional[Union[int, list]] = 30
    workclass: Optional[Union[str, list]] = 'Self-emp-inc'
    fnlgt: Optional[Union[int, list]] = 77516
    education: Optional[Union[str, list]] = 'Bachelors'
    education_num: Optional[Union[int, list]] = Field(10, alias='education-num')
    marital_status: Optional[Union[str, list]] = Field('Never-married', alias='marital-status')
    occupation: Optional[Union[str, list]] = 'Exec-managerial'
    relationship: Optional[Union[str, list]] = 'Not-in-family'
    race: Optional[Union[str, list]] = 'Black'
    sex: Optional[Union[str, list]] = 'Female'
    capital_gain: Optional[Union[int, list]] = Field(6084, alias='capital-gain')
    capital_loss: Optional[Union[int, list]] = Field(0, alias='capital-loss')
    hours_per_week: Optional[Union[int, list]] = Field(40, alias='hours-per-week')
    native_country: Optional[Union[str, list]] = Field('Italy', alias='native-country')

model = load_model("./src/model/model.pkl")
encoder = load_model("./src/model/encoder.pkl")
lb = load_model("./src/model/lb.pkl")

@app.get("/")
def home():
    msg = """Welcome to the 3rd project of the ML Devops Engineer nanodegree
    developed by Mohamed Mejri
    """
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



