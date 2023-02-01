import pytest
import src.train.ml.data as md

@pytest.fixture
def data():
    return md.load_data_s3("./src/data")

def test_load_data_s3(data):
    try:
        df = data
    except BaseException:
        raise "error test_load_data_s3"