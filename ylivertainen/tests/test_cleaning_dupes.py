import pytest
import pandas as pd

from ylivertainen.cleaning import YlivertainenDataCleaningSurg


@pytest.fixture
def make_project():
    def _make_project(df: pd.DataFrame) -> YlivertainenDataCleaningSurg:
        project = YlivertainenDataCleaningSurg.__new__(YlivertainenDataCleaningSurg)
        project.csvs = []
        project.df = df.copy()
        return project
    return _make_project