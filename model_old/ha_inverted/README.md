# HA - Historical average model

Basic long-term forecasting model. Could be used as baseline model. This model compute the mean (or median) of the observation based on one particular feature of the date.

#### Example
 * Data: 1 year, 1 time-series, 1 observation every 30 minutes
 * Feature of the data: time-step (1 observation every 30 minutes: {24h*2}= {1-48}), day of the week (Monday to Sunday: {1-7})
 * Observation data file: csv file with observation every 30 minutes over 1 year
 * Exogenous data file: csv file with exogenous features (day of the week and time-step every 30 minutes) 
 
####Workflow
   1) Edit configuration files
      * config_fit.yaml
        * observation_data_path: csv file path with the observation (header: 'Datetime', 'time_series_id1','time_series_id2', ...)
        * exogenous_data_path: csv file path with the exogenous data (header: 'Datetime', 'feature_id1','feature_id2', ...)
        * features: selected features of exogenous file
        * time_series: selected time-series to learn of observation file
        * model_name: name of the model (e.g. 'my_new_model')
        * path_to_save: path of the directory in which the model will be saved
        * start_date: format 'YYYY-MM-dd hh:mm:ss', starting date of the learning dataset
        * end_date: format 'YYYY-MM-dd hh:mm:ss', ending date of the learning dataset  
    
      * config_predict.yaml
        * observation_data_path: None or csv file path with the dates to predict (header: 'Datetime', whatever)
        * exogenous_data_path: csv file path with the exogenous data (header: 'Datetime', 'feature_id1','feature_id2', ...)
        * path_to_save_prediction: path of the directory in which the prediction will be saved
        * start_date: format 'YYYY-MM-dd hh:mm:ss', starting date of the prediction
        * end_date: format 'YYYY-MM-dd hh:mm:ss', ending date of the prediction
        * model_path: saved model file path (here pkl file)
     
   2) Fit and save the model  
    ```
    # python ha_fit.py --config config_fit.yaml
    ```  
    
   3) Load and predict with the model on old or new data   
    ```
    # python ha_predict.py --config config_predict.yaml
    ```  