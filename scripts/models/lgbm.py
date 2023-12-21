import optuna
from optuna.integration import LightGBMPruningCallback

from darts.models import LightGBMModel
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics import mae
from lightgbm import early_stopping

from .utils import get_categorical_future_covariates

import numpy as np
import logging

QUANTILES = [0.05, 0.5, 0.95]

def build_fit(
    y_train, 
    y_val, 
    pc_train, 
    fc_train, 
    pc_val, 
    fc_val,
    params,
    settings
    ):

    esc = early_stopping(settings['patience'], verbose=True)
    callbacks = [esc]

    if 'LighGBMPruningCallback' in params.keys():
        pruner = params['LighGBMPruningCallback']
        callbacks.append(pruner)

    model = LightGBMModel(
        lags=24*7,
        lags_past_covariates=24*7 if pc_train is not None else None,
        lags_future_covariates=(24*7, 24) if fc_train is not None else None,
        likelihood='quantile',
        quantiles=QUANTILES,
        num_leaves=params['num_leaves'], 
        learning_rate=params['learning_rate'], 
        n_estimators=params['n_estimators'],
        subsample_for_bin=params['subsample_for_bin'], 
        min_child_samples=params['min_child_samples'], 
        subsample=params['subsample'],
        random_state=settings['random_state'],
        categorical_future_covariates=get_categorical_future_covariates() if fc_train is not None else None,
        output_chunk_length=24
    )

    # train the model
    model.fit(
        series=y_train,
        future_covariates=fc_train if model.supports_future_covariates else None, 
        past_covariates=pc_train if model.supports_past_covariates else None,
        val_series=y_val,
        val_future_covariates=fc_val if model.supports_future_covariates else None,
        val_past_covariates=pc_val if model.supports_past_covariates else None,
        callbacks=callbacks
    )

    return model


class Objective(object):
    def __init__(self, y_train, y_val, pc_train, fc_train, pc_val, fc_val, kwargs):
        self.y_train = y_train
        self.y_val = y_val
        self.pc_train = pc_train
        self.fc_train = fc_train
        self.pc_val = pc_val
        self.fc_val = fc_val
        self.kwargs = kwargs

    def __call__(self, trial):

        params = {}

        params['num_leaves'] = trial.suggest_int('num_leaves', 15, 186)
        params['learning_rate'] = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 100)
        params['subsample_for_bin'] = trial.suggest_int('subsample_for_bin', 200_000, 200_000)
        params['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 120)
        params['subsample'] = trial.suggest_float('subsample', 0.1, 1.00)
        params['lags'] = trial.suggest_int('lags', 24, 24*14, step=24)
        #params['LighGBMPruningCallback'] = LightGBMPruningCallback(trial, metric="quantile")

        # build and train the DeepAR model with these hyper-parameters:
        model = build_fit(
            y_train=self.y_train,
            y_val=self.y_val,
            pc_train=self.pc_train,
            pc_val=self.pc_val,
            fc_train=self.fc_train,
            fc_val=self.fc_val,
            params=params,
            settings=self.kwargs
        )

        pc = self.pc_train.append(self.pc_val) if self.pc_train is not None else None
        fc = self.fc_train.append(self.fc_val) if self.fc_train is not None else None

        error = model.backtest(
                series=self.y_train.append(self.y_val),
                future_covariates=fc if model.supports_future_covariates else None,
                past_covariates=pc if model.supports_past_covariates else None,
                retrain=False,
                start=self.y_val.start_time(),
                stride=24,
                metric=mae,
                forecast_horizon=24,
                num_samples=100
            )

        return error


def get_model(y_train, y_val, pc_train, fc_train, pc_val, fc_val, optimize, **kwargs):
    """
    """
    default_params = {
        'num_leaves' : 31,
        'learning_rate' : 0.1,
        'n_estimators' : 100,
        'subsample_for_bin' : 200_000,
        'min_child_samples' : 20,
        'subsample' : 1.0,
        'lags' : 24*7
    }

    if optimize:
        logging.info('Starting hyperparameter optimisation as requested')
        logging.info(f'HPO method: OPTUNA with timeout: {kwargs["timeout"]}')
        
        objective = Objective(y_train, y_val, pc_train, fc_train, pc_val, fc_val, kwargs)
        study = optuna.create_study(direction="minimize")
        study.enqueue_trial(default_params)
        study.optimize(
            objective, 
            timeout=kwargs['timeout'], 
            n_trials=kwargs['n_trials']
            )

        print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
        params = study.best_trial.params
    else:
        logging.info("Skipping HPO per request or as unnecessary")
        params = default_params
        study = None
    
    logging.info("Fitting the model for the first time")
    model = build_fit(
            y_train=y_train,
            y_val=y_val,
            pc_train=pc_train,
            pc_val=pc_val,
            fc_train=fc_train,
            fc_val=fc_val,
            params=params,
            settings=kwargs
        )
    
    return model, study

