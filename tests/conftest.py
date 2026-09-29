import pytest

from annanas.spark import get_spark


@pytest.fixture(scope="session")
def spark():
    session = get_spark("tests", cores=2, memory="2g")
    yield session
    session.stop()
