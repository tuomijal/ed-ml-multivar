#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyprojroot import here

from darts.models import LightGBMModel, TFTModel
from darts.explainability.shap_explainer import ShapExplainer
from darts.explainability.tft_explainer import TFTExplainer

import torch

plt.style.use('default')
mpl.rcParams['figure.dpi'] = 300


# In[ ]:


TARGET = 'occ'
MODEL = 'tft'
FS = 'a'
HPO = 0
H = 24


# In[ ]:


inpath = here() / f'data/processed/models/{TARGET}-{MODEL}-{FS}-{HPO}.pkl'

model = TFTModel.load(str(inpath), map_location='cpu')
model.to_cpu()

pc = model.past_covariate_series

if FS=='a':
    fc = model.future_covariate_series
    start = fc.pd_dataframe().index[0]
    end = fc.pd_dataframe().index[-25]

    ts = model.training_series.slice(start, end)
    
    explainer = TFTExplainer(model, 
                         background_series=ts,
                         background_past_covariates=pc,
                         background_future_covariates=fc)
else:
    explainer = TFTExplainer(model)


# In[ ]:


result = explainer.explain()


# In[ ]:


explainer.plot_attention(result, plot_type='all')
plt.savefig(here() / 'plots/importance-tft.jpg', dpi=300, bbox_inches='tight')
