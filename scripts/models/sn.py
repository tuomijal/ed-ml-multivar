#!/usr/bin/env python
# coding: utf-8

from darts.models import NaiveSeasonal


def get_model(y_train, y_val, pc_train, fc_train, pc_val, fc_val, optimize, **kwargs):
     """
     """
     model = NaiveSeasonal(K=24)
     study = None
     return model, study
