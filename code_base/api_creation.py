import numpy as np
import pandas as pd
import re
from flask import Flask, request, jsonify
app = Flask(__name__)
import pickle

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import json
from config import Config

class ApiCreation:
    def __init__(self,model_pickle_path,data_path):
        self.loaded_model = pickle.load(open(model_pickle_path, 'rb'))
        self.data_path = data_path
        
    def data_attributes(self):
        self.data = pd.read_csv(self.data_path)
        self.data = self.data[['share_close', 'Volume', 'nifty50_close','sentiment_score']]
        self.nobs = 4
        self.data_train, self.data_test = self.data[0:-self.nobs], self.data[-self.nobs:]
    
    def invert_transformation(self,df_train, df_forecast, second_diff=False):
        """Revert back the differencing to get the forecast to original scale."""
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:        
            # Roll back 2nd Diff
            if second_diff:
                df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_1d'].cumsum()
            # Roll back 1st Diff
            df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
        return df_fc
    
    def predict(self,steps):
        self.data_attributes()
        results = self.loaded_model.fit( ic='aic')
        pred = results.forecast(results.y, steps=steps)
        df_forecast = pd.DataFrame(pred, index=self.data.index[-self.nobs:], columns=[i+'_1d' for i in self.data.columns[:4]])
        df_results = self.invert_transformation(self.data_train, df_forecast, second_diff=True)
        return list(df_results['share_close_forecast'][:3])

model_pickle_path = Config().model_config['model_save_path']
data_path = Config().model_config['data_path']
api = ApiCreation(model_pickle_path,data_path)

@app.route("/", methods=['POST', 'GET'])
def forecast():
    try:
        request_json = request.get_json()
        steps = request_json['steps']
        print('steps',steps)
    except Exception as e:
        raise e
    forecast = api.predict(steps)
    print('forecast',forecast)
    result = {'forecast':forecast}
    return jsonify(result)

def main():
    app.run(host="0.0.0.0", port=5005, debug=False,threaded=False)
    
if __name__ == "__main__":
    main()        