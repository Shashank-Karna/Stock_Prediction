import math
from operator import index
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

plt.style.use("fivethirtyeight")


def lstm_model(stock_name):
    df = web.DataReader(
        stock_name, data_source="yahoo", start="2012-01-01", end="2022-04-18"
    )

    # Create a new dataframe with only close column
    data = df.filter(["Close"])
    dataset = data.values  # converting to numpy array
    # get the no of rows to train the model on
    training_data_len = math.ceil(len(dataset) * 0.8)
    # training_data_len

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # scaled_data

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
    # x_train.shape

    # Reshape the data
    x_train = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1], 1)
    )  # (1345, 60, 1)
    # x_train.shape

    # Build the LSTM model

    model = Sequential()
    model.add(
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1))
    )  # (60, 1)

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
    # rmse

    # Plot the data
    train = data[0:training_data_len]
    valid = data[training_data_len:]
    valid["Predictions"] = predictions

    # Convert data to csv for exporting
    df.to_csv(r"" + stock_name[:-3] + "/" + stock_name[:-3] + "_df.csv", index=False)
    train.to_csv(
        r"" + stock_name[:-3] + "/" + stock_name[:-3] + "_train.csv", index=False
    )
    valid.to_csv(
        r"" + stock_name[:-3] + "/" + stock_name[:-3] + "_prediction.csv", index=False
    )

    return


stock_name = input("Enter the name of the stock:\t")
lstm_model(stock_name)
