from models import (arimax, deepar, hwam, hwdm, hwmm, lgbm, nbeats, sn, tft, utils)

def test_model_imports():
	assert 1==1

def test_utils():
	cols = utils.get_categorical_past_covariates()
	print(cols)
	assert type(cols[0]) == str