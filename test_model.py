import os
import pytest
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from src.train.ml.model import save_model



@pytest.fixture
def model():
    return RandomForestClassifier()

@pytest.fixture
def encoder():
    return OneHotEncoder()


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
