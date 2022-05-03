import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import streamlit as st
import warnings

plt.style.use("fivethirtyeight")


warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)

df = web.DataReader(
    "COALINDIA.NS", data_source="yahoo", start="2012-01-01", end="2022-04-18"
)

df

# visualise the closing price
plt.figure(figsize=(16, 8))
plt.title("Close Price History")
plt.plot(df["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price INR", fontsize=18)
plt.show()

# Create a new dataframe with only close column
data = df.filter(["Close"])
dataset = data.values  # converting to numpy array
# get the no of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)
training_data_len

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

# Create the training dataset
# Create the scaled training dataset

train_data = scaled_data[0:training_data_len, :]
# Split the data into X_train and Y_train data sets

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60 : i, 0])
    y_train.append(train_data[i, 0])
#     if i<= 60:
#         print(x_train)
#         print(y_train)
#         print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape


# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # (1345, 60, 1)
x_train.shape

# Build the LSTM model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))  # (60, 1)

model.add(
    LSTM(
        50,
        return_sequences=False,
    )
)
model.add(Dense(25))
model.add(Dense(1))

# Compile the model

model.compile(optimizer="adam", loss="mean_squared_error")

# train the model

model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set

# Create a new array containing scaled values from index 1345 to end

test_data = scaled_data[training_data_len - 60 :, :]

# create the datasets x_test and y_test

x_test = []
y_test = dataset[training_data_len:, :]  # this will contain the actual values

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60 : i, 0])


# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the datra
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# get the models predicted price values

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(
    predictions
)  # we want predictions to contain the same values as y_test dataset


# Get the root mean squared error(RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse

# Plot the data
train = data[0:training_data_len]
valid = data[training_data_len:]
valid["Predictions"] = predictions

# Visulaize the data
plt.figure(figsize=(16, 8))
plt.title("Model")
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price INR", fontsize=18)
plt.plot(train["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Train", "Val", "Predictions"], loc="lower right")
plt.show()

# Show the valid and predicted prices

valid

# try and predict closing price at apr 20 2022
quote = web.DataReader(
    "COALINDIA.NS", data_source="yahoo", start="2012-01-01", end="2022-04-19"
)
# create new dataframe

new_df = quote.filter(["Close"])

# get last 60day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values


# scale the data
last_60_days_scaled = scaler.transform(last_60_days)

x_test1 = []
x_test1.append(last_60_days_scaled)

# convert x_test to numpy array
x_test1 = np.array(x_test1)

# reshape
x_test1 = np.reshape(x_test1, (x_test1.shape[0], x_test1.shape[1], 1))
# print(x_test)

# print(x_test)
# get the predicted scaled price
pred_price = model.predict(x_test1)

# undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

quote2 = web.DataReader(
    "COALINDIA.NS", data_source="yahoo", start="2022-04-20", end="2022-04-20"
)
print(quote2["Close"])

dataa = web.DataReader(
    "HINDALCO.NS", data_source="yahoo", start="2012-01-01", end="2022-04-19"
)

close = dataa.filter(["Close"])

# get last 60day closing price values and convert the dataframe to an array
last_60_days = close[-60:].values

# scale the data
last_60_days_scaled = scaler.transform(last_60_days)

x_input = []
x_input.append(last_60_days_scaled)

# convert x_input to numpy array
x_input = np.array(x_input)

# reshape

x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

pred_price = model.predict(x_input)

# undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
# print(x_input.shape)

# temp_input=list(x_input)

# temp_input=temp_input[0].tolist()
# print(x_input)
# print(temp_input)

close2 = close.tail(60)

# x_test

from numpy import array

lst_output = []
n_steps = 60
i = 0
while i < 30:

    if len(temp_input) > 60:
        x_input = np.array(temp_input[1:], dtype=object).astype("float32")
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        #         print(x_input)
        #         print(yhat)
        temp_input.append(yhat[0].tolist())
        temp_input = temp_input[1:]

        # print(temp_input)

        lst_output.append(yhat[0].tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        #         print(x_input)
        #         print(yhat)
        temp_input.append(yhat[0].tolist())
        lst_output.append(yhat[0].tolist())
        i = i + 1

lst_output

lst_output = scaler.inverse_transform(lst_output)

print(lst_output)

listtt

# Plot the data
# train = data[0: training_data_len]
# valid = data[training_data_len: ]
# valid['Predictions'] = predictions
# close2['next'] = lst_output

# #Visulaize the data
# plt.figure(figsize=(20,8))
# plt.title('Model')
# plt.xlabel('Date', fontsize = 18)
# plt.ylabel('Close Price INR', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close','Predictions']])
# plt.plot(close2['next'])

# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

dataa2 = web.DataReader(
    "HINDUNILVR.NS", data_source="yahoo", start="2012-01-01", end="2022-04-18"
)
dataa2 = dataa2["Close"][-100:]
dataa2 = dataa2.tolist()

list1 = dataa2 + lst_output


plt.figure(figsize=(20, 8))
plt.plot(dataa2)
plt.plot(lst_output)
plt.show()


# quote5 = web.DataReader('ITC.NS', data_source='yahoo', start='2022-01-10', end='2022-04-10')
# #create new dataframe

# new_df = quote5.filter(['Close'])

# #get last 60day closing price values and convert the dataframe to an array
# last_60_days = new_df[-60: ].values

# #scale the data
# last_60_days_scaled = scaler.transform(last_60_days)

# x_test = []
# x_test.append(last_60_days_scaled)

# #convert x_test to numpy array
# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# # x_test
# temp_input=list(x_test)
# temp_input=temp_input[0].tolist()
# temp_input = temp_input[0].values
# tenp_input


# from numpy import array

# lst_output=[]
# n_steps=60
# i=0
# while(i<30):

#     if(len(temp_input)>60):
#         #print(temp_input)
#         x_test=np.array(temp_input[1:])
# #         print("{} day input {}".format(i,x_input))
#         x_test=x_test.reshape(1,-1)
#         x_test = x_test.reshape((1, n_steps, 1))
#         #print(x_input)
#         yhat = model.predict(x_test, verbose=0)
# #         print("{} day output {}".format(i,yhat))
#         temp_input.extend(yhat[0].tolist())
#         temp_input=temp_input[1:]
#         #print(temp_input)
#         lst_output.extend(yhat.tolist())
#         i=i+1
#     else:
#         x_test = x_test.reshape((1, n_steps,1))
#         yhat = model.predict(x_test, verbose=0)
#         print(yhat[0])
#         temp_input.extend(yhat[0].tolist())
#         print(len(temp_input))
#         lst_output.extend(yhat.tolist())
#         i=i+1
