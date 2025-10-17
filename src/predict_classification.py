import pandas as pd
import numpy as np


# Feature list used for training in the notebook
FEATURE_COLS = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'model']


def prepare_features_for_prediction(df_preprocessed: pd.DataFrame, current_machine_id: int):
    """
    Filters the preprocessed data and selects the features needed for the RF model.
    """
    # 1. Filter for the specific machine
    machine_data = df_preprocessed[df_preprocessed['machineID'] == current_machine_id].copy()

    # 2. Select the exact features used for training
    X_current = machine_data[FEATURE_COLS]

    # Ensure 'model' is numerical codes if somehow still object
    if 'model' in X_current.columns and X_current['model'].dtype == 'object':
        X_current['model'] = X_current['model'].astype('category').cat.codes

    return X_current, machine_data['datetime']


def predict_risk(model, X_current: pd.DataFrame):
    """
    Uses the loaded classification model to predict the probability of failure
    for the given feature set (X_current).
    """
    if X_current.empty:
        return np.array([]), 0.0

    # Probability of the positive class (failure == 1)
    probabilities = model.predict_proba(X_current)[:, 1]

    # Highest probability over the period
    max_risk = probabilities.max() * 100 if probabilities.size > 0 else 0.0

    return probabilities, max_risk


