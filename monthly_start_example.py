# this shows monthly start - beginning of the month example
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
df = pd.read_csv("/Users/raja/data/AirPassengers.csv")
df["Month"]=pd.to_datetime(df["Month"])
df.columns = ["ds","y"]
model = Prophet(seasonality_mode ="multiplicative")
model.fit(df)
future = model.make_future_dataframe(periods=365*5,freq ="MS")
forecast = model.predict(future)
fig=model.plot(forecast)
plt.show()
