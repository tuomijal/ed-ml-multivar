#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from pyprojroot import here

from darts import TimeSeries
from darts.dataprocessing.transformers.scaler import Scaler
from darts.metrics import mse
from sklearn.preprocessing import MinMaxScaler
import logging

from utils import (get_data, get_y, get_x, save, get_n_epochs)
from ntrials import ntrials

from models import deepar, lgbm, nbeats, tft
from models import sn, arimax, hwmm, hwdm, hwam

## Likely to be moved to separate module
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CLI interface to run a specific test.')
    # Define the positional arguments
    parser.add_argument('target', 
                        choices=['occ'], help='Target to be tested. Currently only occ is supported.', 
                        metavar='target')
    parser.add_argument('model', 
                        choices=['deepar', 'lgbm', 'nbeats', 'tft', 'sn', 'arimax', 'hwmm', 'hwdm', 'hwam'], 
                        help='Model to be tested. Choose from: deepar, lgbm, nbeats, tft, sn, arimax, hwmm, hwdm, hwam.',
                        metavar='model')
    parser.add_argument('featureset', 
                        choices=['a', 'u'], help='Featureset to use as an input. Select \'a\' or \'u\'.',
                        metavar='featureset')
    parser.add_argument('hpo_indicator', 
                        choices=[0, 1], 
                        type=int, 
                        help='Indicator to either perform HPO (1) or skip it (0).',
                        metavar='hpo_indicator')

    parser.add_argument('-t', '--timeout', 
                        help="Timeout for hyperparameter optimisation in seconds", 
                        type=int,
                        metavar='')
    parser.add_argument('-p', '--patience', 
                        help="Early stopping callback patience value", 
                        type=int,
                        metavar='')
    parser.add_argument('-e', '--epochs', 
                        help="Max number of epochs", 
                        type=int,
                        metavar='')
    parser.add_argument('-n', '--name', 
                        help="Additional identifier for persistence",
                        metavar='')
    parser.add_argument('-V', '--valstart', 
                        help="Validation start date",
                        metavar='')
    parser.add_argument('-T', '--teststart', 
                        help="Test start date",
                        metavar='')
    parser.add_argument('-S', '--headstart', 
                        help="Data start date",
                        metavar='')
    parser.add_argument('-E', '--tailstop', 
                        help="Data end date",
                        metavar='')
    parser.add_argument('-r', '--randomstate', 
                        help="Random state for reproducibility", 
                        type=int,
                        metavar='')

    args = parser.parse_args()

    MODEL = args.model
    TARGET = args.target
    FEATURESET = args.featureset
    HPO = args.hpo_indicator
    TESTNAME = f'{TARGET}-{MODEL}-{FEATURESET}'

    # SETTINGS
    TIMEOUT = args.timeout if args.timeout else int(os.environ.get('TIMEOUT', 86400))
    PATIENCE = args.patience if args.patience else int(os.environ.get('PATIENCE', 10))
    EPOCHS = args.epochs if args.epochs else ntrials[TESTNAME]['epochs'] if TESTNAME in ntrials.keys() else 100
    NAME = args.name if args.name else None
    RANDOMSTATE = args.randomstate if args.randomstate else os.environ.get('RANDOMSTATE', 0)
    RANDOMSTATE = int(RANDOMSTATE) if RANDOMSTATE is not None else None
    FIRSTSPLIT = args.valstart if args.valstart else os.environ.get('FIRSTSPLIT', '20180101')
    SECONDSPLIT = args.teststart if args.teststart else os.environ.get('SECONDSPLIT', '20180620')
    FIRSTSPLIT = pd.Timestamp(FIRSTSPLIT)
    SECONDSPLIT = pd.Timestamp(SECONDSPLIT)

    # Tämä on pakko miettiä jotenkin paremmin
    NTRIALS = ntrials[TESTNAME]['n_trials'] if TESTNAME in ntrials.keys() else 10

    SETTINGS = {
        'FIRSTSPLIT' : FIRSTSPLIT,
        'SECONDSPLIT' : SECONDSPLIT
    }

    # ## Preprocess
    logging.basicConfig(
        format='%(asctime)s - %(message)s', 
        level=logging.INFO,
        filename=here() / f"logs/logger/{TARGET}-{MODEL}-{FEATURESET}-{HPO}.log",
        filemode='w'
        )

    logging.info('Starting test.')
    logging.info(f'MODEL: {MODEL} TARGET: {TARGET} FEATURESET: {FEATURESET} HPO: {HPO} TIMEOUT: {TIMEOUT} RANDOMSTATE: {RANDOMSTATE} EPOCHS: {EPOCHS} PATIENCE: {PATIENCE} NTRIALS: {NTRIALS}')


    data = get_data(TARGET, MODEL)

    # Get time series
    y = get_y(data, TARGET)
    pc, fc = get_x(data, FEATURESET)

    # Scaling and splitting y
    if MODEL!='lgbm':
        scaler = MinMaxScaler(feature_range=(0, 1))
        y_transformer = Scaler(scaler)
        y = y_transformer.fit_transform(y)

        # Scaling and splitting X 
        x_transformer = Scaler(scaler)

        pc = x_transformer.fit_transform(pc) if pc is not None else None
        fc = x_transformer.fit_transform(fc) if fc is not None else None

    # Convert past_covariates into lagged future_covariates for deepar
    if MODEL=='deepar' and FEATURESET=='a':
        pc = pc.shift(24)
        fc = fc.slice_intersect(pc)
        pc = pc.slice_intersect(fc)
        fc = fc.concatenate(pc, axis=1)
        y = y.slice_intersect(fc)

    pc_train_val, _ = pc.split_before(SECONDSPLIT) if pc is not None else (None, None)
    fc_train_val, _ = fc.split_before(SECONDSPLIT) if fc is not None else (None, None)

    pc_train, pc_val = pc_train_val.split_before(FIRSTSPLIT) if pc is not None else (None, None)
    fc_train, fc_val = fc_train_val.split_before(FIRSTSPLIT) if fc is not None else (None, None)

    y_train_val, y_test = y.split_before(SECONDSPLIT)
    y_train, y_val = y_train_val.split_before(FIRSTSPLIT)

    model, study = eval(MODEL).get_model(
        y_train=y_train, 
        y_val=y_val, 
        pc_train=pc_train, 
        fc_train=fc_train, 
        pc_val=pc_val, 
        fc_val=fc_val, 
        optimize=int(HPO), 
        timeout=TIMEOUT, 
        patience=PATIENCE, 
        epochs=EPOCHS, 
        random_state=RANDOMSTATE,
        n_trials=NTRIALS
        )

    # Adjust testbench
    if MODEL in ['hwam', 'hwmm', 'hwdm', 'arimax', 'sn']:
        train_length = 24*7
        retrain = True
    else:
        train_length = y_train_val.n_timesteps
        retrain = 30

    # Determine number of epochs in retrain phase based on initial ESC result
    if MODEL in ['deepar', 'tft', 'nbeats']:
        model.model_params['pl_trainer_kwargs']['callbacks'] = []
        model.model_params['n_epochs'] = get_n_epochs(model)

    logging.info('Starting backtesting on the test set')

    y_pred = model.historical_forecasts(
        series=y,
        future_covariates=fc if model.supports_future_covariates else None,
        past_covariates=pc if model.supports_past_covariates else None,
        num_samples=1 if MODEL in ['sn'] else 100,
        start=SECONDSPLIT,
        forecast_horizon=24,
        retrain=retrain,
        train_length=train_length,
        last_points_only=False,
        verbose=True,
        stride=24,
        overlap_end=True
        )

    if MODEL!='lgbm':
        y_pred = [y_transformer.inverse_transform(x) for x in y_pred]

    save(model_name=MODEL,
         featureset_name=FEATURESET,
         target_name=TARGET,
         hpo=HPO,
         y_pred=y_pred,
         model=model,
         study=study,
         settings=SETTINGS
         )

    logging.info('Results persisted')