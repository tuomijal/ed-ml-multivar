#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from pyprojroot import here
import sys

conversion = {
    'occ' : 'Target:Occupancy',
    'arr' : 'arrivals',
    'los' : 'length_of_stay'
}

TARGET = sys.argv[1]
INPATH = here() / f'data/raw/data.csv'
OUTPATH = here() / f'data/processed/true_matrices/'

data = pd.read_csv(INPATH, index_col='datetime', parse_dates=True)
y = data[conversion[TARGET]]
trues = np.ones((len(y), 24)) * np.nan

for n, i in enumerate(y[:-24]):
    y_true = y[n:n+24]
    trues[n] = y_true

columns = [f"t+{x+1}" for x in range(trues.shape[1])]
result = pd.DataFrame(data=trues, 
                      index=y.index,
                      columns=columns)
result = result.iloc[::24]

OUTPATH.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPATH / f'{TARGET}.csv')