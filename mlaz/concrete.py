"""Physics-informed features for the chapter 12 concrete-strength capstone.

This lives in an importable module (not the notebook) on purpose: a pipeline saved
with joblib stores functions *by reference*, so ``joblib.load`` of the capstone model
works anywhere ``mlaz`` is installed.
"""
import numpy as np
import pandas as pd

CONCRETE_COMPONENTS = ["cement", "slag", "ash", "water", "superplastic", "coarseagg", "fineagg"]
CONCRETE_INPUTS = CONCRETE_COMPONENTS + ["age"]


def add_physics_features(X):
    """Append physics-informed ratios to the raw mix design (all inputs in kg/m³, age in days)."""
    X = pd.DataFrame(X, columns=CONCRETE_INPUTS).copy()
    binder = X["cement"] + X["slag"] + X["ash"]
    X["w_c"] = X["water"] / X["cement"]
    X["binder"] = binder
    X["w_b"] = X["water"] / binder
    X["log_age"] = np.log(X["age"])
    X["agg_binder"] = (X["coarseagg"] + X["fineagg"]) / binder
    X["sp_binder"] = X["superplastic"] / binder
    X["scm_frac"] = (X["slag"] + X["ash"]) / binder
    return X
