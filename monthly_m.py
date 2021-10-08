# this shows monthly start - multiplicative - no prediction
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
df = pd.read_csv("/Users/raja/data/AirPassengers.csv")
df["Month"]=pd.to_datetime(df["Month"])
df.columns = ["ds","y"]
model = Prophet(seasonality_mode ="multiplicative", yearly_seasonality=4)
model.fit(df)
forecast = model.predict()
fig=model.plot(forecast)
plt.show()
fig_a2 = model.plot_components(forecast)
plt.show()
