# Rain-ARIMA-Model

ARIMA models provide another approach to time series forecasting. Exponential smoothing and ARIMA models are the two most widely used approaches to time series forecasting, and provide complementary approaches to the problem. While exponential smoothing models are based on a description of the trend and seasonality in the data, ARIMA models aim to describe the autocorrelations in the data.

Before we introduce ARIMA models, we must first discuss the concept of stationarity and the technique of differencing time series.

import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")
sns.set()

#import csv file using panda library
df = pd.read_csv("rain.csv")
df

end_date = "1905-06-21"
model_ret_sarimax = SARIMAX(df.x[1:], order = (3,0,4), seasonal_order = (3,0,2,5))
results_ret_sarimax = model_ret_sarimax.fit()

df_pred_sarimax = results_ret_sarimax.predict(start = start_date, end = end_date) 

df_pred_sarimax[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.x[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actual", size = 24)
plt.show()

# Forecast for the next 10 pwriods.
pred = model_auto.predict(start=(len), n_period=10)
pd.DataFrame(pred, columns = ['prediction'])

#apply for the model_auto
model_auto = auto_arima(df.x[1:], m = 5, max_p = 5, max_q = 5, max_P = 5, max_Q = 5)
