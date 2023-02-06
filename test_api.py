import pytest
from fastapi.testclient import TestClient
from main import app
from src.train.ml.data import load_data

client = TestClient(app)

@pytest.fixture()
def data():
    return load_data("./src/data/census_cleaned.csv")

@pytest.fixture()
def zero_class(data):
    return data[data["salary"] == "<=50K"].iloc[0]

@pytest.fixture()
def one_class(data):
    return data[data["salary"] == ">50K"].iloc[0]

def test_get():
    response = client.get("/")
    assert response.status_code == 200, "Couldn't establish connection"
    assert response.json() == {"greetings": "Welcome to the 3rd project of the ML Devops Engineer nanodegree developed by Mohamed Mejri"}

def test_post_zero_class(zero_class):
    response = client.post("/api", json=zero_class.to_dict())
    assert response.status_code == 200, "ERROR while testing zero_class"
    assert response.json() == {"predictions": "['<=50K']"}, "ERROR: didn't predict zero class"

def test_post_one_class(one_class):
    response = client.post("/api", json=one_class.to_dict())
    assert response.status_code == 200, "ERROR while testing one_class"
    assert response.json() == {"predictions": "['>50K']"}, "ERROR: didn't predict one class"    
