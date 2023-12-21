import optuna
from optuna.integration import PyTorchLightningPruningCallback

from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics import mse, mae

import torch
from pytorch_lightning.callbacks import EarlyStopping

import numpy as np
from .utils import get_model_name, get_early_stopper, get_pl_trainer_kwargs

import logging

QUANTILES = [0.05, 0.5, 0.95]
INPUT_CHUNK_LENGTH = 24*7
OUTPUT_CHUNK_LENGTH = 24

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
    """
    """

    early_stopper = get_early_stopper(settings['patience'])
    pl_trainer_kwargs = get_pl_trainer_kwargs([early_stopper])
    model_name = get_model_name('nbeats_model')
    num_workers = 0

    model = NBEATSModel(
        input_chunk_length=params['input_chunk_length'],
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        num_stacks=params['num_stacks'],
        num_blocks=params['num_blocks'],
        num_layers=params['num_layers'],
        layer_widths=params['layer_widths'],
        batch_size=params['batch_size'],
        optimizer_kwargs={"lr": params['lr']},
        model_name=model_name,
        likelihood=QuantileRegression(quantiles=QUANTILES),
        pl_trainer_kwargs=pl_trainer_kwargs,
        force_reset=True,
        save_checkpoints=True,
        random_state=settings['random_state'],
        n_epochs=settings['epochs']
    )

    # train the model
    model.fit(
        series=y_train,
        future_covariates=fc_train if model.supports_future_covariates else None, 
        past_covariates=pc_train if model.supports_past_covariates else None,
        val_series=y_val,
        val_future_covariates=fc_val if model.supports_future_covariates else None,
        val_past_covariates=pc_val if model.supports_past_covariates else None,
        num_loader_workers=num_workers
    )

    # reload best model over course of training
    model = NBEATSModel.load_from_checkpoint(model_name, best=True)

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

        params['num_stacks'] = trial.suggest_int('num_stacks', 15, 60)
        params['num_blocks'] = trial.suggest_int('num_blocks', 1, 2)
        params['num_layers'] = trial.suggest_int('num_layers', 2, 8)
        params['layer_widths'] = trial.suggest_categorical('layer_widths', [128, 256, 512])
        params['batch_size'] = trial.suggest_categorical("batch_size", [16, 32])
        params['lr'] = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3, 1e-2])
        params['input_chunk_length'] = trial.suggest_int('input_chunk_length', 24, 24*14, step=24)

        # build and train the model with these hyper-parameters:
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

        # Evaluate how good it is on the validation set
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
        'num_stacks' : 30,
        'num_blocks' : 1,
        'num_layers' : 4,
        'layer_widths' : 256,
        'batch_size' : 32,
        'lr' : 0.001,
        'input_chunk_length' : 24*7
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