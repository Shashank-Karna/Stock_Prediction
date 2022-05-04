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


def plot_closing_price(df):
    # visualise the closing price
    st.write("The closing prices of the stock.")
    plt.figure(figsize=(16, 8))
    plt.title("Close Price History")
    plt.plot(df["Date"], df["Close"])
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
    plt.plot(train["Date"], train["Close"])
    plt.plot(valid["Date"], valid[["Close", "Predictions"]])
    plt.legend(["Train", "Val", "Predictions"], loc="lower right")
    st.pyplot()


def plot_30(d1, d2):
    st.write("Predictions for next 30 days")
    plt.title("Close Price for next 30 days")
    plt.ylabel("Close Price INR", fontsize=18)
    plt.plot(d1)
    plt.plot(d2["Numbers"], d2["0"])
    # plt.legend(["Train", "Val", "Predictions"], loc="lower right")
    st.pyplot()


def plot_final(final_graph):
    plt.figure(figsize=(16, 8))

    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.title("Close Price after 30 days")
    plt.axhline(
        y=final_graph[len(final_graph) - 1],
        color="red",
        linestyle=":",
        label="NEXT 30D: {0}".format(
            round(float(*final_graph[len(final_graph) - 1]), 2)
        ),
    )
    plt.plot(final_graph)
    plt.legend()
    st.pyplot()


nav = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction of Stock Price for next 30 days",
        "Close Price Prediciton for today",
        "Recommended stocks",
        "About us",
    ],
    index=0,
)

try:

    if nav == "Home":
        st.title("INVESTMENT RECOMMENDATION SYSTEM")

        st.header("Invest today for a better tomorrow.")

        st.write(
            "A stock market is a platform for trading of a company’s stocks and derivatives at an agreed price. Supply and demand of shares drive the stock market. In any country the stock market is one of the most emerging sectors. Nowadays, many people are indirectly or directly related to this sector. Therefore, it becomes essential to know about market trends. Thus, with the development of the stock market, people are interested in forecasting stock prices."
        )

        image = Image.open("test_stock.jpg")
        st.image(image, caption="")

        st.subheader("Our Objective")

        st.write(
            "Of India's 1.36 billion people, only about 3.7 per cent invest in the stock market. To spread financial literacy and make analysis easy for better investments."
        )
        st.write(
            "Investment is a complex process, especially understanding the stock market, mutual funds, etc.So, the main objective is to bridge the awareness gap and let users plan and decide what stocks are best to invest their money."
        )
        st.write(
            "Also, to analyze risk factors and help people invest their hard earned money better."
        )

        st.subheader("Machine Learning Model - LSTM")

        st.write(
            "Long Short Term Memory is a kind of recurrent neural network (RNN). LSTM can by default retain the information for a long period of time. It is used for processing, predicting, and classifying on the basis of time-series data."
        )

        st.write("Training is done under the following metrics and functions:")

        st.write("""- Number of layers = 4 (two hidden layers)""")
        st.write("""- Loss function = Root Mean Square Error""")
        st.write("""- Optimizer = Adam""")
        st.write("""- Epoch = 100""")

    elif nav == "Prediction of Stock Price for next 30 days":

        stock = st.selectbox(
            "Enter the name of the stock: ", ("GAIL", "BPCL", "ONGC", "CIPLA", "ITC")
        )

        train = pd.read_csv(stock + "/" + stock + "_train.csv")
        valid = pd.read_csv(stock + "/" + stock + "_prediction.csv")

        plot_predictions(train, valid)

        st.write(valid)

        f = open(stock + "/" + "pred.txt", "r")
        lines = f.readlines()

        st.markdown("**5th May 2022**")
        st.text("    Predicted Price: " + lines[2])

        d1 = pd.read_csv(stock + "/" + stock + "_100.csv")
        d2 = pd.read_csv(stock + "/" + stock + "_30.csv")

        plot_30(d1, d2)

        final = pd.read_csv(stock + "/" + stock + "_final.csv")

        plot_final(final)

    elif nav == "Close Price Prediciton for today":
        stock_list = ["GAIL", "BPCL", "ONGC", "CIPLA", "ITC"]
        for stock in stock_list:
            f = open(stock + "/" + "pred.txt", "r")
            lines = f.readlines()
            st.subheader(stock)
            st.markdown("**4th May 2022**")
            st.text(
                """   Predicted Price:: {0}
   Actual Close Price: {1}""".format(
                    lines[0], lines[1]
                )
            )

            st.markdown("**5th May 2022**")
            st.text("    Predicted Price: " + lines[2])

    elif nav == "Recommended stocks":
        avg_inc_10y = [
            19.716022195962758,
            21.406139636123463,
            6.119890773694583,
            21.139191716757697,
            -6.056964676487557,
            30.608961946762044,
            20.522642396492667,
            33.998616423070246,
            5.47611494366027,
            14.16692400771014,
            22.642841682640537,
            23.75461639876866,
            18.525326535911407,
            11.254367301648143,
            22.164762430739582,
            1.0782703577490962,
        ]
        avg_inc_5y = [
            9.486513509473767,
            0.9719351756197684,
            -1.9087238724276445,
            26.076924246725998,
            -12.550032762899864,
            30.861984007768854,
            35.677714221906115,
            32.765446720279016,
            -1.3967261875930141,
            14.079724378815897,
            35.99683962147297,
            32.88653079717078,
            27.636024200936276,
            26.011140540327858,
            26.16855428852932,
            -1.6003915160816935,
        ]
        symbols = [
            "AXISBANK",
            "BPCL",
            "ITC",
            "HINDUNILVR",
            "COALINDIA",
            "ASIANPAINT",
            "WIPRO",
            "TECHM",
            "HEROMOTOCO",
            "CIPLA",
            "RELIANCE",
            "HINDALCO",
            "NESTLEIND",
            "BHARTIARTL",
            "TCS",
            "ONGC",
        ]

        st.subheader(
            "Current Nifty50 Stocks with average increase greater than 15% in last 10 years: "
        )

        stocks_10y = []
        stocks_5y = []
        for i in range(len(avg_inc_10y)):
            if round(avg_inc_10y[i]) > 15:
                stocks_10y.append(symbols[i])
                st.write(symbols[i], ":", avg_inc_10y[i])

        st.subheader(
            "\nCurrent Nifty50 Stocks with average increase greater than 15% in last 5 years: "
        )

        for i in range(len(avg_inc_5y)):
            if round(avg_inc_5y[i]) > 15:
                stocks_5y.append(symbols[i])
                st.write(symbols[i], ":", avg_inc_5y[i])

        stocks_10y = set(stocks_10y)
        stocks_5y = set(stocks_5y)

        common = stocks_10y.intersection(stocks_5y)

        st.subheader("\nStocks Recommended on the basis of above results: ")
        for i in common:
            st.write(i)

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
