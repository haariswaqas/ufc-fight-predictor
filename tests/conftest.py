import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ufc import load_config  # noqa: E402
from ufc.clean import build_clean  # noqa: E402
from ufc.features import build_all  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def clean(cfg):
    return build_clean(cfg)


@pytest.fixture(scope="session")
def built(clean, cfg):
    """(bouts, prefight, features) for the full data."""
    return build_all(clean, cfg)
