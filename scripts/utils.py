import pandas as pd
import numpy as np
import os
import re

from darts import TimeSeries
from darts.utils.model_selection import train_test_split

from pyprojroot import here
import joblib

from sklearn.preprocessing import LabelEncoder

# TODO: This feels hacky, is there a better way? 
def get_n_epochs(model):
    """
    Check logs for how many epochs were used to train the
    initial model
    """
    path = model.load_ckpt_path
    n_epochs = re.search('checkpoints/best-epoch=(.*)-val_loss', path)
    return int(n_epochs.group(1)) + 1


def preprocess(data, model):
    if model!='lgbm':
        # Process holidays
        holiday_names = pd.get_dummies(data['Calendar:Holiday_name'], prefix='Holiday_name', prefix_sep=':')
        holiday_names = holiday_names.iloc[:,1:]
        data = data.drop(columns='Calendar:Holiday_name')
        data = pd.concat([data, holiday_names], axis=1)

        # Process other calendar variables
        hour = pd.get_dummies(data['Calendar:Hour'], prefix='Hour', prefix_sep=':')
        weekday = pd.get_dummies(data['Calendar:Weekday'], prefix='Weekday', prefix_sep=':')
        month = pd.get_dummies(data['Calendar:Month'], prefix='Month', prefix_sep=':')

        data = data.drop(columns=['Calendar:Hour', 'Calendar:Month', 'Calendar:Weekday'])
        data = pd.concat([data, hour, weekday, month], axis=1)

        # Process slip
        slip = pd.get_dummies(data['Weather:Slip'], prefix='Weather_Slip', prefix_sep=':')
        data = data.drop(columns='Weather:Slip')
        data = pd.concat([data, slip], axis=1)
    elif model=='lgbm':
        le = LabelEncoder()
        # Process holidays
        data['Calendar:Holiday_name'] = le.fit_transform(data['Calendar:Holiday_name'])
    else:
        raise ValueError('Model not supported')
        
    # Fill nans
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    return data


def get_data(target_name, model):
    # Import data
    data = pd.read_csv(here() / "data/interim/data.csv", 
                       index_col='datetime',
                       parse_dates=True,
                       low_memory=False)

    data = data["2017":]
    
    data = data.asfreq('h')
    data = preprocess(data, model)

    return data


def get_y(data, target_name):
    cd = {
        'occ' : 'Target:Occupancy',
        'arr' : 'Target:Arrivals'
    }
    target_name = cd[target_name]
    series = TimeSeries.from_series(data[target_name])
    return series


def get_x(data, featureset):
    # Masks
    traffic = data.columns.str.startswith('Traffic')
    beds = data.columns.str.startswith('Beds')
    google = data.columns.str.startswith('Trends')
    website = data.columns.str.startswith('Website_visits')
    ta = data.columns.str.startswith('TA')

    weather = data.columns.str.startswith('Weather')
    calendar = data.columns.str.startswith('Calendar')
    public_event = data.columns.str.startswith('Events')
    
    hours = data.columns.str.startswith('Hour')
    weekdays = data.columns.str.startswith('Weekday')
    months = data.columns.str.startswith('Month')
    holidays = data.columns.str.startswith('Holiday_name')

    if featureset=='u':
        past_covariates = None
        future_covariates = None
    if featureset=='a':
        past_cov_mask =  traffic | beds | google | website | ta
        past_covariates = data.loc[:,past_cov_mask]
        future_cov_mask = months | weekdays | hours | calendar | holidays | public_event | weather
        future_covariates = data.loc[:,future_cov_mask]
    
    if past_covariates is not None:
        past_covariates = past_covariates.fillna(0)
        past_covariates = TimeSeries.from_dataframe(past_covariates)
    if future_covariates is not None:
        future_covariates = future_covariates.fillna(0)
        future_covariates = TimeSeries.from_dataframe(future_covariates)

    return past_covariates, future_covariates


def to_pred_matrix(ts_list, quantile=None):
    """
    Converts multihorizontal TimeSeries results received from any 
    <darts.model>.historical_forecast into properly indexed prediction matrix.
    """
    matrix = list()

    for ts in ts_list:
        if quantile:
            vector = ts.quantile_df(quantile).iloc[:,0]
        else:
            vector = ts.pd_series()

        vector.name = vector.index[0]
        vector.index = [f"t+{x+1}" for x in range(len(vector))] 
        matrix.append(vector)

    matrix = pd.concat(matrix, axis=1).T
    matrix.index.name = 'datetime'
    matrix = matrix.round(2)

    return matrix


def save(
    model_name, 
    featureset_name, 
    target_name,
    hpo,
    y_pred, 
    model,
    study=None,
    quantiles=[.05, .50, .95],
    settings=None
    ):
    """
    Perstists prediction matrix and model binary
    """
    unique_name = f'{target_name}-{model_name}-{featureset_name.lower()}-{hpo}'

    rootpath = here('data/processed/prediction_matrices')

    if y_pred[0].is_probabilistic:
        for quantile in quantiles:
            matrix = to_pred_matrix(y_pred, quantile)
            
            outpath = rootpath / f'{int(quantile*100):02d}'
            outpath.mkdir(parents=True, exist_ok=True)
            matrix.to_csv(outpath / f"{unique_name}.csv")
    
    if y_pred[0].is_deterministic:
        matrix = to_pred_matrix(y_pred)
        outpath = rootpath / f'50'
        outpath.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(outpath / f"{unique_name}.csv")

    # models
    outpath = here('data/processed/models')
    outpath.mkdir(parents=True, exist_ok=True)
    model_path = str(outpath / f'{unique_name}.pkl')
    model.save(model_path)

    if study:
        # studies
        outpath = here('data/processed/studies')
        outpath.mkdir(parents=True, exist_ok=True)
        model_path = str(outpath / f'{unique_name}.pkl')
        joblib.dump(study, model_path)

    if settings:
        outpath = here('logs/settings.pkl')
        joblib.dump(settings, outpath)
