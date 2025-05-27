# TSA_EXP9EX.NO.09 A project on Time series analysis on weather forecasting using ARIMA model
Date:
## AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.

## ALGORITHM:
Explore the dataset of weather
Check for stationarity of time series time series plot ACF plot and PACF plot ADF test Transform to stationary: differencing
Determine ARIMA models parameters p, q
Fit the ARIMA model
Make time series predictions
Auto-fit the ARIMA model
Evaluate model predictions
## PROGRAM
```
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the Gold Price data
file_path = 'Gold Price Prediction.csv'
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
series = data['Price Today']

# Perform Augmented Dickey-Fuller test
result = adfuller(series)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Plot the time series, ACF, and PACF, and save as an image
plt.figure(figsize=(12, 12))
plt.subplot(311)
plt.plot(series)
plt.title("Gold Price Series")

plt.subplot(312)
plot_acf(series, ax=plt.gca(), lags=20)
plt.title("ACF Plot")

plt.subplot(313)
plot_pacf(series, ax=plt.gca(), lags=20)
plt.title("PACF Plot")
plt.tight_layout()
plt.savefig('gold_acf_pacf.png')
plt.show()

# Check if differencing is needed based on the p-value
if result[1] > 0.05:
    series_diff = series.diff().dropna()
else:
    series_diff = series

# Fit an ARIMA model and display the summary
model = ARIMA(series_diff, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# Forecast the next 10 days
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)

# Create a new index for the forecasted values
forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

# Plot the original series and forecast, then save as an image
plt.figure(figsize=(12, 6))
plt.plot(series, label="Original Series")
plt.plot(forecast_index, forecast, label="Forecast", color='red')
plt.title("ARIMA Model Forecast on Gold Price")
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.legend()
plt.savefig('gold_forecast.png')
plt.show()
```
## OUTPUT:

![image](https://github.com/user-attachments/assets/c25379cc-29bb-473b-89c0-7f693eb1963e)

![image](https://github.com/user-attachments/assets/f3c8f971-1035-43c7-8875-618b785486f9)


![image](https://github.com/user-attachments/assets/a3d6d3f3-6095-4d8e-aa64-786cd37d2fd9)

![image](https://github.com/user-attachments/assets/d953f1f6-d413-4c28-95b7-995a1d59e25a)



## RESULT:
Thus the program run successfully based on the ARIMA model using python.
