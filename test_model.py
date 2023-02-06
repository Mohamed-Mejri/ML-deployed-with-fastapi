import os
import pytest
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from src.train.ml.model import save_model, load_model


def test_save_model(model):
    path = Path("./model.pkl")
    clf = model
    try:
        save_model(clf, name="model.pkl" ,path="./")
    except:
        print("ERROR: unable to save model")
    
    assert path.is_file(), "ERROR: unable to save model"
    os.remove("./model.pkl")

def test_save_encoder(encoder):
    path = Path("./encoder.pkl")
    enc = encoder
    try:
        save_model(enc, "encoder.pkl", "./")
    except: 
        print("ERROR: unable to save encoder")
    
    assert path.is_file(), "ERROR: unable to save encoder"
    os.remove("./encoder.pkl")

def test_load_models():
    try:
        encoder = load_model("src/model/encoder.pkl")
        model = load_model("src/model/model.pkl")
        lb = load_model("src/model/lb.pkl")
    except:
        print("ERROR while loading models/encoders")
    finally:
        assert isinstance(encoder, OneHotEncoder), "Unexpected encoder type"
        assert isinstance(model, RandomForestClassifier), "Unexpected model type"
        assert isinstance(lb, LabelEncoder), "Unexpected label encoder type"
