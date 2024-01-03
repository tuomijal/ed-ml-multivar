# Forecasting emergency department occupancy with advanced machine learning models and multivariable input

https://doi.org/10.1016/j.ijforecast.2023.12.002

Authors:

**Jalmari Tuominen (1), Eetu Pulkkinen (1), Jaakko Peltonen (2), Juho Kanniainen (2), Niku Oksala (1,3), Ari Palomäki (1,4) and Antti Roine (1)**

1) *Faculty of Medicine and Health Technology, Tampere University*
2) *Faculty of Information Technology and Communication Sciences, Tampere University*
3) *Centre for Vascular Surgery and Interventional Radiology, Tampere University Hospital*
4) *Kanta-Häme Central Hospital, Hämeenlinna, Finland*

## Abstract
Emergency department (ED) crowding is a significant threat to patient safety and it has been repeatedly associated with increased mortality. Forecasting future service demand has the potential to improve patient outcomes. Despite active research on the subject, proposed forecasting models have become outdated due to quick influx of advanced machine learning models (ML) and because the amount of multivariable input data has been limited. In this study, we document the performance of a set of advanced ML models in forecasting ED occupancy 24 hours ahead. We use electronic health record data from a large, combined ED with an extensive set of explanatory variables, including the availability of beds in catchment area hospitals, traffic data from local observation stations, weather variables and more. We show that DeepAR, N-BEATS, TFT and LightGBM all outperform traditional benchmarks with up to 15% improvement. The inclusion of the explanatory variables enhances the performance of TFT and DeepAR but fails to significantly improve the performance of LightGBM. To the best of our knowledge, this is the first study to extensively document the superiority of ML over statistical benchmarks in the context of ED forecasting.


## Usage

This repository contains code, data and raw text of the [article](https://doi.org/10.1016/j.ijforecast.2023.12.002). The repository uses GNU Make to manage the complete pipeline from raw data to manuscript PDF. The code works both locally and in a SLURM cluster. Configuration details and caveats of these two options are provided below.


### Local

You can excecute the code on your local machine by following these steps:

1. Clone the repository
2. Create virtual environment

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

:warning: Note for M1 users. LightGBM does not support arm64 so you will have to use Rosetta. However, using Rosetta will break polars and you will have to explicitly remove it by calling `pip uninstall polars`. This will effectively revert to `polars-lts-cpu` that is included in the requirements.

3. Option 1: Run make

Makefile has been split into two parts because of the long time it will take to backtest the models and because downstream dependencies will break if the code is excecuted in SLURM cluster:

```
make preddata   # backtest models
make            # generate tables and plots and compile the manuscript
```

Note however that on your local machine `make preddata` this will backtest all models in serial which will take a considerable amount of time regardless of the power of your machine. In most cases this option serves more as a documentation of how the results are generated than as a viable option to recreate the results from scratch.

4. Option 2: Use CLI interface

We also provide a terminal client interface to reproduce the specific results you are interested in. It works by calling `python scripts/train.py` with the following arguments:

```
usage: train.py [-h] [-t] [-p] [-e] [-n] [-V] [-T] [-S] [-E] [-r]
                target model featureset hpo_indicator

CLI interface to run a specific test.

positional arguments:
  target               Target to be tested. Currently only occ is supported.
  model                Model to be tested. Choose from: deepar, lgbm, nbeats,
                       tft, sn, arimax, hwmm, hwdm, hwam.
  featureset           Featureset to use as an input. Select 'a' or 'u'.
  hpo_indicator        Indicator to either perform HPO (1) or skip it (0).

options:
  -h, --help           show this help message and exit
  -t , --timeout       Timeout for hyperparameter optimisation in seconds
  -p , --patience      Early stopping callback patience value
  -e , --epochs        Max number of epochs
  -n , --name          Additional identifier for persistence
  -V , --valstart      Validation start date
  -T , --teststart     Test start date
  -S , --headstart     Data start date
  -E , --tailstop      Data end date
  -r , --randomstate   Random state for reproducibility
```

For example, this would backtest $LightGBM_{U}$ model without hyperparameter optimisation:

```
python scripts/train.py occ lgbm u 0
```

### SLURM

You can excecute the code in a SLURM cluster by following these steps:

1. Login to SLURM using your credentials
2. Clone this repository
3. Create the virtual environment

```
module load pytorch
export PYTHONUSERBASE=/projappl/<your_project>/my-python-env
pip install --user darts lightgbm optuna pyprojroot papermill
```

4. Set ACCOUNT environment variable

Set account environment variable where ID is the identifier you project.

```
export ACCOUNT=project_{ID}
``` 

5. Configure settings (Optional)

Timeout and patience can be adjusted by defining environment variables e.g.:

```
export TIMEOUT=3600
export PATIENCE=4
```

6. Run make

```
make preddata
```

Make will automatically create separate batch jobs for each model. After the batch jobs are finished, we recommend donwloading the results to your local machine and then call `make` again which will run the notebooks to generate tables and plots and finally compiles the manuscript.


## Data

The data and it's sources is described in detail in the manuscript. The available columns is provided below for clarity:

|   No | Name                             |
|-----:|:---------------------------------|
|    1 | Beds:HC_01                       |
|    2 | Beds:HC_02                       |
|    3 | Beds:HC_03                       |
|    4 | Beds:HC_04                       |
|    5 | Beds:HC_05                       |
|    6 | Beds:HC_06                       |
|    7 | Beds:HC_07                       |
|    8 | Beds:HC_08                       |
|    9 | Beds:HC_09                       |
|   10 | Beds:HC_10                       |
|   11 | Beds:HC_11_(Ward_01)             |
|   12 | Beds:HC_11_(Ward_02)             |
|   13 | Beds:HC_11_(Ward_03)             |
|   14 | Beds:HC_12                       |
|   15 | Beds:HC_13                       |
|   16 | Beds:HC_14                       |
|   17 | Beds:HC_15                       |
|   18 | Beds:RH_A_(Ward_01)              |
|   19 | Beds:RH_A_(Ward_02)              |
|   20 | Beds:RH_A_(Ward_03)              |
|   21 | Beds:RH_A_(Ward_04)              |
|   22 | Beds:RH_A_(Ward_05)              |
|   23 | Beds:RH_A_(Ward_06)              |
|   24 | Beds:RH_A_(Ward_07)              |
|   25 | Beds:RH_A_(Ward_08)              |
|   26 | Beds:RH_A_(Ward_09)              |
|   27 | Beds:RH_A_(Ward_10)              |
|   28 | Beds:RH_A_(Ward_11)              |
|   29 | Beds:RH_B_(Ward_01)              |
|   30 | Beds:RH_B_(Ward_02)              |
|   31 | Beds:UH_(ED_ward)                |
|   32 | Beds:UH_(Ward_01)                |
|   33 | Beds:UH_(Ward_02)                |
|   34 | Calendar:Holiday_name            |
|   35 | Calendar:Holiday_t+0             |
|   36 | Calendar:Holiday_t+1             |
|   37 | Calendar:Holiday_t+2             |
|   38 | Calendar:Holiday_t+3             |
|   39 | Calendar:Holiday_t-1             |
|   40 | Calendar:Holiday_t-2             |
|   41 | Calendar:Holiday_t-3             |
|   42 | Calendar:Hour                    |
|   43 | Calendar:Is_working_day          |
|   44 | Calendar:Month                   |
|   45 | Calendar:Preceding_holidays      |
|   46 | Calendar:Weekday                 |
|   47 | Events                           |
|   48 | TA:Occupancy-AO                  |
|   49 | TA:Occupancy-ATAN                |
|   50 | TA:Occupancy-CMO                 |
|   51 | TA:Occupancy-COS                 |
|   52 | TA:Occupancy-COSH                |
|   53 | TA:Occupancy-EXP                 |
|   54 | TA:Occupancy-FLOOR               |
|   55 | TA:Occupancy-HT_DCPERIOD         |
|   56 | TA:Occupancy-HT_DCPHASE          |
|   57 | TA:Occupancy-HT_TRENDLINE        |
|   58 | TA:Occupancy-HT_TRENDMODE        |
|   59 | TA:Occupancy-KAMA                |
|   60 | TA:Occupancy-LINEARREG           |
|   61 | TA:Occupancy-LINEARREG_ANGLE     |
|   62 | TA:Occupancy-LINEARREG_INTERCEPT |
|   63 | TA:Occupancy-LINEARREG_SLOPE     |
|   64 | TA:Occupancy-MAX                 |
|   65 | TA:Occupancy-MAXINDEX            |
|   66 | TA:Occupancy-MIDPOINT            |
|   67 | TA:Occupancy-MIN                 |
|   68 | TA:Occupancy-MININDEX            |
|   69 | TA:Occupancy-MOM                 |
|   70 | TA:Occupancy-PO                  |
|   71 | TA:Occupancy-ROC                 |
|   72 | TA:Occupancy-ROCR100             |
|   73 | TA:Occupancy-RSI                 |
|   74 | TA:Occupancy-SIN                 |
|   75 | TA:Occupancy-SINH                |
|   76 | TA:Occupancy-SMA                 |
|   77 | TA:Occupancy-SQRT                |
|   78 | TA:Occupancy-STDDEV              |
|   79 | TA:Occupancy-SUM                 |
|   80 | TA:Occupancy-TAN                 |
|   81 | TA:Occupancy-TANH                |
|   82 | TA:Occupancy-TRIMA               |
|   83 | TA:Occupancy-VAR                 |
|   84 | TA:Occupancy-WMA                 |
|   85 | Target:Occupancy                 |
|   86 | Traffic:203_to_Hämeenkyrö        |
|   87 | Traffic:203_to_Ylöjärvi          |
|   88 | Traffic:204_to_Ikaalinen         |
|   89 | Traffic:204_to_Parkano           |
|   90 | Traffic:210_to_Huittinen         |
|   91 | Traffic:210_to_Vammala           |
|   92 | Traffic:401_to_Hämeenlinna       |
|   93 | Traffic:401_to_Tampere           |
|   94 | Traffic:402_to_Hämeenlinna       |
|   95 | Traffic:402_to_Tampere           |
|   96 | Traffic:404_to_Orivesi           |
|   97 | Traffic:404_to_Tampere           |
|   98 | Traffic:406_to_Pori              |
|   99 | Traffic:406_to_Tampere           |
|  100 | Traffic:409_to_Ruovesi           |
|  101 | Traffic:409_to_Virrat            |
|  102 | Traffic:421_to_Hämeenkyrö        |
|  103 | Traffic:421_to_Tampere           |
|  104 | Traffic:422_to_Tampere           |
|  105 | Traffic:422_to_Urjala            |
|  106 | Traffic:431_to_Kangasala         |
|  107 | Traffic:431_to_Tampere           |
|  108 | Traffic:433_to_Hämeenlinna       |
|  109 | Traffic:433_to_Tampere           |
|  110 | Traffic:435_to_Orivesi           |
|  111 | Traffic:435_to_Tampere           |
|  112 | Traffic:436_to_Nokia             |
|  113 | Traffic:436_to_Tampere           |
|  114 | Traffic:438_to_Nokia             |
|  115 | Traffic:438_to_Tampere           |
|  116 | Traffic:439_to_Kangasala         |
|  117 | Traffic:439_to_Tampere           |
|  118 | Traffic:440_to_Hämeenlinna       |
|  119 | Traffic:440_to_Tampere           |
|  120 | Traffic:443_to_Nokia             |
|  121 | Traffic:443_to_Tampere           |
|  122 | Traffic:445_to_Humppila          |
|  123 | Traffic:445_to_Tampere           |
|  124 | Traffic:448_to_Jämsä             |
|  125 | Traffic:448_to_Orivesi           |
|  126 | Traffic:449_to_Nokia             |
|  127 | Traffic:449_to_Tampere           |
|  128 | Traffic:450_to_Nokia             |
|  129 | Traffic:450_to_Ylöjärvi          |
|  130 | Traffic:451_to_Orivesi           |
|  131 | Traffic:451_to_Tampere           |
|  132 | Traffic:452_to_Nokia             |
|  133 | Traffic:452_to_Tampere           |
|  134 | Traffic:453_to_Nokia             |
|  135 | Traffic:453_to_Tampere           |
|  136 | Traffic:455_to_Tampere           |
|  137 | Traffic:455_to_Ylöjärvi          |
|  138 | Traffic:456_to_Tampere           |
|  139 | Traffic:456_to_Ylöjärvi          |
|  140 | Traffic:457_to_Kangasala         |
|  141 | Traffic:457_to_Nokia             |
|  142 | Traffic:458_to_Kangasala         |
|  143 | Traffic:458_to_Tampere           |
|  144 | Traffic:460_to_Nokia             |
|  145 | Traffic:460_to_Tampere           |
|  146 | Traffic:462_to_Nokia             |
|  147 | Traffic:462_to_Tampere           |
|  148 | Traffic:463_to_Nokia             |
|  149 | Traffic:463_to_Tampere           |
|  150 | Traffic:464_to_Hämeenlinna       |
|  151 | Traffic:464_to_Tampere           |
|  152 | Trends:Acuta                     |
|  153 | Weather:Air_pressure             |
|  154 | Weather:Air_temp                 |
|  155 | Weather:Cloud_count              |
|  156 | Weather:Day_air_temp_max         |
|  157 | Weather:Day_air_temp_min         |
|  158 | Weather:Dew_point_temp           |
|  159 | Weather:Rain_intensity           |
|  160 | Weather:Rel_hum                  |
|  161 | Weather:Slip                     |
|  162 | Weather:Snow_depth               |
|  163 | Weather:Visibility               |
|  164 | Website_visits:Domain_1          |
|  165 | Website_visits:Domain_1_(ER)     |
|  166 | Website_visits:Domain_1_(EV)     |
|  167 | Website_visits:Domain_2          |
