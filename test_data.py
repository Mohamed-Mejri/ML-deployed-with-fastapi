import os
import pytest
from pathlib import Path
from src.train.ml.data import load_data_s3
from src.train.constants import COLUMNS


def test_load_data_s3(data):
    path = Path("./tmp.csv")

    try:
        df = data
    except BaseException:
        raise "error test_load_data_s3"

    assert (df.columns == COLUMNS).any(), "unexpected columns in data"
    assert path.is_file(), "data wasn't found in directory"
    