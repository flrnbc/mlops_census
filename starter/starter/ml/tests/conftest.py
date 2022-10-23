""" Fixtures to share among some tests """
import os
import pytest

import starter.ml.data as data

@pytest.fixture(scope="session")
def data_path():
    """Ensures that we can run pytest from everywhere
    in the terminal.
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    return f"{file_dir}/test_data.csv"


@pytest.fixture(scope="session")
def test_df(data_path):
    return data.load_data(data_path)
