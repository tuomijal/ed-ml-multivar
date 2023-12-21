from datetime import datetime
from uuid import uuid4
from pyprojroot import here
import pandas as pd

import torch
from pytorch_lightning.callbacks import EarlyStopping

from utils import get_data

def get_model_name(name):
    """
    """
    now = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    model_name = now + '_' + name + '_' + str(uuid4())
    return model_name


def get_early_stopper(patience):
    """
    """
    early_stopper = EarlyStopping("val_loss", patience=patience, verbose=True)
    return early_stopper


def get_pl_trainer_kwargs(callbacks):
    """
    """
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "devices": "auto",
            "callbacks": callbacks,
        }
    else:
        pl_trainer_kwargs = {
            "callbacks": callbacks,
            "devices": "auto"
            }

    return pl_trainer_kwargs


def get_categorical_future_covariates():
    """
    """
    data = get_data('occupancy', 'lgbm')
    cols = data.columns

    calendar = cols.str.startswith('Calendar')
    slip = cols.str.startswith('Weather:Slip')
    cols = calendar | slip
    cols = data.columns[cols].values.tolist()
    
    return cols