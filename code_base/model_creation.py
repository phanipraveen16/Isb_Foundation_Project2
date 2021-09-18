import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
import pickle
from statsmodels.tsa.stattools import adfuller
from config import Config
from sklearn import metrics

class Model:
    def __init__(self):
        'constructor of the class'
        self.config = Config().model_config
        
    def model(self):
        'model function that will be overridden'
        pass
    
class RelianceIndustries(Model):
    'Class for generating reliance industries'
    def adf_test(self,ts, signif=0.05):
        dftest = adfuller(ts, autolag='AIC')
        adf = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags','# Observations'])
        for key,value in dftest[4].items():
            adf['Critical Value (%s)'%key] = value
        print (adf)
        p = adf['p-value']
        if p <= signif:
            print(f" Series is Stationary")
            return 'Stationary'
        else:
            print(f" Series is Non-Stationary")
            return 'Non-Stationary'
    
    # inverting transformation
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
    
    def model(self):
        'function to get xgboost model with random search cv'
        data_path = self.config['data_path']
        model_save_path = self.config['model_save_path']
        data = pd.read_csv(data_path)
        data = data[['share_close', 'Volume', 'nifty50_close','sentiment_score']]
        nobs = 4
        data_train, data_test = data[0:-nobs], data[-nobs:]
        status = self.adf_test(data_train["share_close"])
        if(status == 'Non-Stationary'):
            data_differenced = data_train.diff().dropna()
            status = self.adf_test(data_differenced["share_close"])
            print('status for differenced',status)
        else:
            data_differenced = data_train
        model = VAR(data_differenced)
        results = model.fit( ic='aic')
        lag_order = results.k_ar
        results.forecast(data.values[-lag_order:], 5)
        results.plot_forecast(20)
        pred = results.forecast(results.y, steps=nobs)
        df_forecast = pd.DataFrame(pred, index=data.index[-nobs:], columns=[i+'_1d' for i in data.columns])

        # show inverted results in a dataframe
        df_results = self.invert_transformation(data_train, df_forecast, second_diff=True)     
        rmse = np.sqrt(metrics.mean_squared_error(data_test['share_close'][:3], df_results['share_close_forecast'][:3]))
        print('rmse',rmse)
        pickle.dump(model, open(model_save_path, 'wb')) 
        print('model saved')

class ModelFactory:
    'Factory class for getting model'
    def get_model(self, type_of_model):
        'function to get model'
        if 'reliance_industries' in type_of_model.lower():
            return RelianceIndustries()
        else:
            raise ValueError('invalid type of model')

model_obj =  ModelFactory().get_model(Config().model_config['model_name'])
model_obj.model()                
        
    
    


