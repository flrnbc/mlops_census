""" Fixtures to share among some tests """
from pathlib import Path

import pytest
import starter.starter.ml.data as data


@pytest.fixture(scope="session")
def test_df():
    current_dir = Path(__file__).parent
    test_dir = current_dir / "starter" / "starter" / "ml" / "tests"
    return data.load_data(test_dir/"test_data.csv")
