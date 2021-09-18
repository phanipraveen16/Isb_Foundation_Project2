# Isb_Foundation_Project2

### Stock value prediction model using multivariate time series forecasting using sentiment of stock and previous share price.
#### Sentiment is found on news articles extracted through news_api org and share data is extracted through yfinance python package
#### Data preprocessing is completely handled using spark for handling scalability, code of which can be found at spark_data_handling.ipynb
#### Multivariate forecasting is done using VAR model with help of python statsmodel package
#### Model is deployed on Heroku and prediction can be done by using api end point at https://isb-foundation-project2.herokuapp.com/forecast, the sample code to call api is found at api_call.py in code_base directory.