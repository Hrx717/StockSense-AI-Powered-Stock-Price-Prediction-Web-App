import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

model = load_model('./Stock Prediction Model.keras')


st.header('Predict the Stock Price')

stock = st.text_input('Enter the Stock Name', 'GOOG')
start = '2012-01-01'
end = '2024-12-31'

data = yf.download(stock, start, end)

st.subheader('STOCK DATA')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test  = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price VS MA100')
ma_100_days = data.Close.rolling(100).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100: i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

y_predict = model.predict(x)

y_predict = scaler.inverse_transform(y_predict)
y = scaler.inverse_transform(y.reshape(-1, 1))


st.subheader('Original Price VS Predicted Price')
fig2 = plt.figure(figsize=(8,6))
plt.plot(y_predict, 'r', label="Predicted Price")
plt.plot(y, 'g', label = "Original Price")

plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig2)