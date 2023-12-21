import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

PURPLE = (138/255, 97/255, 178/255)
GREEN = (59/255, 178/255, 154/255)
RED = (206/255, 48/255, 110/255)
YELLOW = (255/255, 226/255, 98/255)
DARK_GREEN = (28/255, 88/255, 77/255)
BLUE = (120/255, 192/255, 232/255)

colordict = {
    "lgbm" : "tab:red",
    'tft' : "tab:green",
    'deepar' : "tab:blue",
    'nbeats' : "tab:orange",
    'sn' : ".9",
    'hwdm' : ".8",
    'hwam' : ".7",
    'arimax' : ".6"
}

name_mask = {
    'deepar' : 'DeepAR',
    'lgbm' : 'LightGBM',
    'arimax' : 'ARIMA',
    'nbeats' : 'N-BEATS',
    'tft' : 'TFT',
    'hwdm' : 'HWDM',
    'hwam' : 'HWAM',
    'sn' : 'SN'
}

def interval_score(
    observations,
    alpha,
    q_dict=None,
    q_left=None,
    q_right=None,
    percent=False,
    check_consistency=True,
    mean=False,
    scaled=False,
    seasonality=None,
):
    """
    Courtesy of adrian-lison
    https://github.com/adrian-lison/interval-scoring/blob/master/scoring.py
    
    Compute interval scores (1) for an array of observations and predicted intervals.
    
    Either a dictionary with the respective (alpha/2) and (1-(alpha/2)) quantiles via q_dict needs to be
    specified or the quantiles need to be specified via q_left and q_right.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alpha : numeric
        Alpha level for (1-alpha) interval.
    q_dict : dict, optional
        Dictionary with predicted quantiles for all instances in `observations`.
    q_left : array_like, optional
        Predicted (alpha/2)-quantiles for all instances in `observations`.
    q_right : array_like, optional
        Predicted (1-(alpha/2))-quantiles for all instances in `observations`.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield a percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total interval scores.
    sharpness : array_like
        Sharpness component of interval scores.
    calibration : array_like
        Calibration component of interval scores.
        
    (1) Gneiting, T. and A. E. Raftery (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102(477), 359–378.    
    """

    if q_dict is None:
        if q_left is None or q_right is None:
            raise ValueError(
                "Either quantile dictionary or left and right quantile must be supplied."
            )
    else:
        if q_left is not None or q_right is not None:
            raise ValueError(
                "Either quantile dictionary OR left and right quantile must be supplied, not both."
            )
        q_left = q_dict.get(alpha / 2)
        if q_left is None:
            raise ValueError(f"Quantile dictionary does not include {alpha/2}-quantile")

        q_right = q_dict.get(1 - (alpha / 2))
        if q_right is None:
            raise ValueError(
                f"Quantile dictionary does not include {1-(alpha/2)}-quantile"
            )

    if check_consistency and np.any(q_left > q_right):
        raise ValueError("Left quantile must be smaller than right quantile.")
        
    if scaled:
        if mean is None:
            raise ValueError("Mean has to be True in order to scale.")
        if seasonality is None:
            raise ValueError("Seasonality has to be set to scale")

    sharpness = q_right - q_left
    calibration = (
        (
            np.clip(q_left - observations, a_min=0, a_max=None)
            + np.clip(observations - q_right, a_min=0, a_max=None)
        )
        * 2
        / alpha
    )
    if percent:
        sharpness = sharpness / np.abs(observations)
        calibration = calibration / np.abs(observations)
        
    total = sharpness + calibration
    
    if mean:
        mis = np.mean(total)
        return mis
    
    if scaled:
        msis = mis / np.mean(np.abs(observations[m:] - observations[:-m]))
        return msis
    
    return total, sharpness, calibration

def downsample(df, T):
    """
    """
    n_samples = (df.y_true >= T).value_counts()[1]
    crowding_false_mask = df.y_true < T
    crowding_true_mask = df.y_true >= T

    a = df[crowding_false_mask].sample(n_samples, random_state=23)
    b = df[crowding_true_mask]

    df = pd.concat([a,b])
    return df

def bootstrap(y_true, y_pred, function, n_boostrap=250):
    """
    """
    bootstrapped_scores = []
    
    for i in range(n_boostrap):
        indices = np.random.randint(0, len(y_pred), len(y_pred))    
        values = function(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(values)
        
    bootstrapped_scores = np.array(bootstrapped_scores)
    sorted_scores = np.sort(bootstrapped_scores)
    
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    
    return confidence_lower, confidence_upper

def bootstrap_auc(y_true, y_pred, T, n_boostrap=250):
    """
    """
    bootstrapped_scores = []
    
    for i in range(n_boostrap):
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        
        fpr, tpr, thresholds = roc_curve(y_true[indices] > T, y_pred[indices], pos_label=1)
        _auc = auc(fpr, tpr)
        bootstrapped_scores.append(_auc)
        
    bootstrapped_scores = np.array(bootstrapped_scores)
    sorted_scores = np.sort(bootstrapped_scores)
    
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    
    return confidence_lower, confidence_upper

def pairwise(df, control, test):
    """
    Performer pairwise test for each column of df compared to
    control series.
    """
    results = {}

    for c in df.columns:
        vector = df[c]
        s, p = test(vector, control)
        results[c] = p

    p_values = pd.Series(results)
    return p_values