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
            st.write("Predictions for Coal India Limited")

            df = pd.read_csv("COALINDIA\COALINDIA_df.csv")

            plot_closing_price(df)

        BPCL = st.checkbox("BPCL.NS")
        if BPCL:
            st.write("Predictions for Bharat Petroleum Corporation Limited")

            df = pd.read_csv("BPCL/BPCL_df.csv")
            plot_closing_price(df)

        ITC = st.checkbox("ITC.NS")
        if ITC:
            st.write("Predictions for ITC Limited")

            df = pd.read_csv("ITC/ITC_df.csv")
            plot_closing_price(df)

        AxisBank = st.checkbox("Axis Bank")
        if AxisBank:
            st.write("Predictions for Axis Bank Limited")

            df = pd.read_csv("AXISBANK/AXISBANK_df.csv")
            plot_closing_price(df)

        GAIL = st.checkbox("GAIL.NS")
        if GAIL:
            st.write("Predictions for GAIL (India) Limited")

            df = pd.read_csv("GAIL/GAIL_df.csv")
            plot_closing_price(df)

    elif nav == "Projection Accuracy":
        st.write(
            "Select the Stock to see the accuracy of the predictions for it over the last 30 days."
        )

        CoalIndia = st.checkbox("COALINDIA.NS")
        if CoalIndia:
            st.write("Predictions for CoalIndia")

            df = pd.read_csv("COALINDIA\COALINDIA_df.csv")
            train = pd.read_csv("COALINDIA\COALINDIA_train.csv")
            valid = pd.read_csv("COALINDIA\COALINDIA_prediction.csv")

            plot_closing_price(df)
            plot_predictions(train, valid)

            st.write(valid)

        BPCL = st.checkbox("BPCL.NS")
        if BPCL:
            st.write("Predictions for Bharat Petroleum Corporation Limited")

            df = pd.read_csv("BPCL/BPCL_df.csv")
            train = pd.read_csv("BPCL/BPCL_train.csv")
            valid = pd.read_csv("BPCL/BPCL_prediction.csv")

            plot_closing_price(df)
            plot_predictions(train, valid)
            st.write(valid)

        ITC = st.checkbox("ITC.NS")
        if ITC:
            st.write("Predictions for ITC Limited")

            df = pd.read_csv("ITC/ITC_df.csv")
            train = pd.read_csv("ITC/ITC_train.csv")
            valid = pd.read_csv("ITC/ITC_prediction.csv")

            plot_closing_price(df)
            plot_predictions(train, valid)
            st.write(valid)

        AxisBank = st.checkbox("Axis Bank")
        if AxisBank:
            st.write("Predictions for Axis Bank Limited")

            df = pd.read_csv("AXISBANK/AXISBANK_df.csv")
            train = pd.read_csv("AXISBANK/AXISBANK_train.csv")
            valid = pd.read_csv("AXISBANK/AXISBANK_prediction.csv")

            plot_closing_price(df)
            plot_predictions(train, valid)
            st.write(valid)

        GAIL = st.checkbox("GAIL.NS")
        if GAIL:
            st.write("Predictions for GAIL (India) Limited")

            df = pd.read_csv("GAIL/GAIL_df.csv")
            train = pd.read_csv("GAIL/GAIL_train.csv")
            valid = pd.read_csv("GAIL/GAIL_prediction.csv")

            plot_closing_price(df)
            plot_predictions(train, valid)
            st.write(valid)

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
