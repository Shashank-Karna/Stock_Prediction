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
from PIL import Image

plt.style.use("fivethirtyeight")

st.set_option("deprecation.showPyplotGlobalUse", False)
warnings.filterwarnings("ignore")


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

    return df, train, valid


def plan_generator(amount):
    st.write("The investment amount is:", amount)


def plot_closing_price(df):
    # visualise the closing price
    st.write("The closing prices of the stock.")
    plt.figure(figsize=(16, 8))
    plt.title("Close Price History")
    plt.plot(df["Close"])
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price INR", fontsize=18)
    st.pyplot()


def plot_predictions(train, valid):
    # Visulaize the data
    st.write("The Predictions vs Real time data")
    plt.figure(figsize=(16, 8))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price INR", fontsize=18)
    plt.plot(train["Close"])
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Train", "Val", "Predictions"], loc="lower right")
    st.pyplot()


nav = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Recommended Investment Plan",
        "Projections for different stocks",
        "Projection Accuracy",
        "About us",
    ],
    index=0,
)

try:

    if nav == "Home":
        st.header("Home")

        st.write("INTRODUCTION AND SOME INFORMATION ABOUT STOCKS")

        image = Image.open("test_stock.jpg")
        st.image(image, caption="RANDOM STOCK IMAGE")

        st.write("SOME INFORMATION ABOUT THE PROJECT")

    elif nav == "Recommended Investment Plan":
        st.write("Here we ask user for amount and give the plan")

        amount = st.number_input(
            "Please enter the amount you would like to invest this month.",
            step=10,
        )

        plan_generator(amount)

    elif nav == "Projections for different stocks":
        st.write(
            "Select the Stock to see the predictions for it over the next 30 days."
        )
        CoalIndia = st.checkbox("COALINDIA.NS")
        if CoalIndia:
            st.write("Predictions for CoalIndia")

            df, train, valid = lstm_model("COALINDIA.NS")
            plot_closing_price(df)

        Stock2 = st.checkbox("Stock2")
        if Stock2:
            st.write("Predictions for Stock2")

            df, train, valid = lstm_model("Stock2")
            plot_closing_price(df)

        Stock3 = st.checkbox("Stock3")
        if Stock3:
            st.write("Predictions for Stock3")

            df, train, valid = lstm_model("Stock3")
            plot_closing_price(df)

    elif nav == "Projection Accuracy":
        st.write(
            "Select the Stock to see the accuracy of the predictions for it over the last 30 days."
        )

        CoalIndia = st.checkbox("COALINDIA.NS")
        if CoalIndia:
            st.write("Predictions for CoalIndia")

            df, train, valid = lstm_model("COALINDIA.NS")
            plot_closing_price(df)
            plot_predictions(train, valid)

            st.write(valid)

        Stock2 = st.checkbox("Stock2")
        if Stock2:
            st.write("Predictions for Stock2")

            df, train, valid = lstm_model("Stock2")
            plot_closing_price(df)
            plot_predictions(train, valid)

        Stock3 = st.checkbox("Stock3")
        if Stock3:
            st.write("Predictions for Stock3")

            df, train, valid = lstm_model("Stock3")
            plot_closing_price(df)
            plot_predictions(train, valid)

    elif nav == "About us":
        st.header("About Us")
        st.write(
            "This project is an initiative towards creating a SIP plan to invest in stocks by getting accurate predictions about the closing prices of the most popular stocks."
        )
        st.write(
            "The idea is to create awareness in the people about investing in Stocks and to help them do so in a safe and comfortable manner."
        )
        st.subheader("Project Created by")
        st.write("Shashank Karna")
        st.write("Akash Khatri")
        st.write("Rishika Lulla")
        st.subheader("Suggestions")
        suggestion = st.text_area(
            "Please feel free to drop some suggestions to help us improve."
        )
        st.subheader("Queries")
        st.write("For any queries contact us via the following details")
        st.write("Email ID: shashankkarna01@gmail.com")

        st.subheader("THANK YOU!!!")

except:
    pass
