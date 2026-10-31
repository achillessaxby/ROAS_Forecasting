import os
from dataclasses import replace

import pytest

from roas.config import DEFAULT_SETTINGS
from roas.data import load_directory
from roas.synthetic import generate

AS_OF = "2025-09-01"


@pytest.fixture(scope="session")
def settings():
    # small models keep the suite fast; backend chosen by CI matrix (sklearn | xgboost)
    return replace(DEFAULT_SETTINGS, backend=os.environ.get("ROAS_TEST_BACKEND", "auto"), n_estimators=120)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("data"), latest=AS_OF)


@pytest.fixture(scope="session")
def games(data_dir):
    return load_directory(data_dir)


@pytest.fixture(scope="session")
def registry_root(tmp_path_factory, data_dir, settings):
    from roas.pipeline import train

    root = tmp_path_factory.mktemp("registry")
    train(data_dir, root, AS_OF, settings=settings, promote=True)
    return root
