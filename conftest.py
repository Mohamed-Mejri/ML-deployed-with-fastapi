import pytest
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from src.train.ml.data import load_data_s3

@pytest.fixture(scope='session')
def data():
    return load_data_s3("tmp.csv")

@pytest.fixture(scope='session')
def model():
    return RandomForestClassifier()

@pytest.fixture(scope='session')
def encoder():
    return OneHotEncoder()

@pytest.fixture()
def zero_class(data):
    return data[data["salary"] == "<=50K"].iloc[0]

@pytest.fixture()
def one_class(data):
    return data[data["salary"] == ">50K"].iloc[0]
    