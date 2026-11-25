"""Datasets used in the documentation."""

from importlib.resources import files

import pandas as pd


def theophylline() -> pd.DataFrame:
    """Theophylline in 12 subjects after one oral dose (Boeckmann, Sheiner and Beal 1994).

    One row per record: ID; AMT, the dose in mg/kg at time 0; TIME in hours; DV, the
    concentration in mg/L, missing on the dose rows; WEIGHT in kg; SEX.
    """
    with files(__name__).joinpath("theophylline.csv").open() as f:
        return pd.read_csv(f, na_values=".")
