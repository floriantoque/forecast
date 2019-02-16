import typing
import os

class Forecast_model:

    def __init__(self, name: str):
        self.infos = {}
        self.infos['name'] = name


    def fit(self, df_observation, df_date):
        return


    def predict(self, list_date):
        return


    def __str__(self):
        return "Model %s" % (self.infos['name'])



