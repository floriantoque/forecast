# Forecast
The 'Forecast' project proposes the implementation of time-series forecasting models. Several type of time-series can be predicted (e.g., crowd in transport network, weather, financial time-series, ...).  This project is built in order to easily learn, save and load each forecasting model and then predict on new data. Three different forecasting models are developed in this project:  
  * (i) Long-term model: Used to predict time-series until one year ahead. The learning step is based on date information. It needs a certain correlation between the time-series pattern and the given date. 
  * (ii) Middle-term model: Used to predict time-series until month, week or day ahead. The learning step is based on the use of exogenous data that can help the prediction (e.g., weather, social network information, events, ...). In general such information are given one month, week or day ahead, that is why, the model is named "middle-term".
  * (iii) Short-term model: Used to predict time-series until one or a few time-step ahead (multi time-step ahead prediction). 

Each model implementation is located in a specific directory:
  * model/ha/: Historical average model (long-term model)
  * ...

  

  
## Requirements

* Python 3.5
  * Pandas
  * Numpy
  * ...
  
For an easier configuration, it is possible to use the docker configuration available in this repo: https://github.com/floriantoque/docker 



## Data

Data is essential to learn the forecasting models. Two types of csv files are used in this project.
#### 1) Observation data
The observation data file contains the numerical observed values for each of the time-series at each time step.
* Header: 'Datetime','id_timeseries1', 'id_timeseries2', etc..
* Values: 'YYYY-MM-dd hh-mm-ss', int or float, int or float, etc..

![](docs/observation_data.png?raw=true "example")

#### 2) Exogenous data
The exogenous data files contain the numerical observed values of exogenous data at each time-step.   

* Header: 'Datetime','id_feature1', 'id_feature2', etc..
* Data : Datetime (format: YYYY-MM-dd hh-mm-ss), values of observation (Int or Float)
 
![](docs/exogenous_data.png?raw=true "example")

**Note** : Each time-step in the observation data file needs to be available in the exogenous data file.


## Forecasting workflow 


#### 1) Go to the model directory 
 * e.g. Historical average model (directory ha)
 
```
cd model/ha/
```




#### 2) Edit configuration files
Configuration files are:
  * config\_optimize.yaml (1): parameters used for the optimization of the model
  * config_fit.yaml: parameters used for the learning step of the model
  * config\_predict.yaml: parameters used for prediction
 
 
(1) some model do not need this file (ha model) 

#### 3) Optimize the model
**Note** : some model do not need this file (ha model)

```
# python <script_optimize.py> --config config_optimize.yaml
```

#### 4) Fit and save model
```
# python <script_fit.py> --config config_fit.yaml
```

#### 5) Optimize the model
```
# python <script_predict.py> --config config_predict.yaml
```

