#divvy_1.py
# this shows monthly start - multiplicative - no prediction
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
df = pd.read_csv("/Users/raja/data/divvy_daily.csv")
# let us only take dates and rides columns ignore the others..
df = df[['date','rides']]
# convert to proper date-time format.
df['date'] = pd.to_datetime(df['date'])
# rename columns for prophet
df.columns = ['ds','y']
# design the model
model = Prophet(seasonality_mode="multiplicative")
# see how well it fits...
model.fit(df)
forecast = model.predict()
fig = model.plot(forecast)
plt.show()
# Now predict the future
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
fig2 = model.plot(forecast)
plt.show()
# shows 3 components :
# 1. trend
# 2. weekly 
# 3. yearly ( as we have atleast 2 years of data)
fig3 = model.plot_components(forecast)
plt.show()