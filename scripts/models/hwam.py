#!/usr/bin/env python
# coding: utf-8

from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode

def get_model(y_train, y_val, pc_train, fc_train, pc_val, fc_val, optimize, **kwargs):
    """
    """
    model = ExponentialSmoothing(
        trend=ModelMode.ADDITIVE,
        damped=False,
        seasonal=SeasonalityMode.ADDITIVE,
        seasonal_periods=24
    )
    study = None
    return model, study