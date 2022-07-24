# -*- coding: utf-8 -*-

# @Date:        2022/06/01
# @Author:      Ch'i YU
# @Last Edited: 2022/07/22 18:00
# @Project Contribution: TBC

"""
Some Text Here

The goal of this project is to ...

Anyway I'm going to name this website after "Ka-Ching!"

"""

# import necessary dependencies

import streamlit as st

import os
import re
import gc
import urllib
import time
import datetime

import pandas as pd
import numpy as np

from keras import models

from plotly import graph_objs as go
from plotly.figure_factory import create_candlestick
from plotly.graph_objects import Line, Marker

from sklearn.preprocessing import MinMaxScaler

# ______________________________________________________________________________________________________________________
# main page

def main():
    """
    Starting `streamlit` execution in a main() function.
    """

    # set page title / layout / sidebar
    st.set_page_config(
        page_title = "Ka-Ching: Visualize your AI-Supported Portfolio!",
        page_icon = ":bar_chart:",
        initial_sidebar_state = "expanded",
        layout = "wide")

    # set page header
    st.header("Ka-Ching: Visualize your AI-Supported Portfolio!")

    # set github badge series
    st.markdown(
        f"""
        [gh]: https://github.com/Ch-i-Yu/Portfolio-Management
        ![GitHub Repo stars](https://img.shields.io/github/stars/Ch-i-YU/Portfolio-Management?style=social)
        ![GitHub Repo forks](https://img.shields.io/github/forks/Ch-i-YU/Portfolio-Management?style=social)
        """
    )

    # set page slogan
    st.markdown(
        """
        > *Powered by **LSTM**, Visualized with **plotly**;* 
        > *Deployed on **Streamlit Cloud**, All Designed just for **YOU**:sparkling_heart:.*
        """
    )

    # set page about / userguide info
    tab_about, tab_plotlytricks, tab_userguide = st.tabs(["About", "Plotly Tricks", "User Guide"])
    with tab_about:
        st.markdown(
            """
            About Content: Hot Work in Progress!
            """
        )
    with tab_plotlytricks:
        st.markdown(
            """
            - Double Click on legends to isolate a trace;
            - Single Click on legends to remove a trace;
            - Repeat your clicks to undo your operations;
            """
        )
    with tab_userguide:
        st.markdown(
            """
            1. Config your portfolio in the **`sidebar`**;
            2. Press the **`button`** below in the **`sidebar`** to start;
            
            Please double check the validity of your portfolio configurations if the **`button`** is disabled.
            """
        )

    # config global variables
    stock_option = ["AAPL", "AMZN", "BRK-B", "GOOG", "JNJ", "JPM",
                    "MSFT", "NVDA", "PG", "TSLA", "V", "WMT", "XOM"]
    risk_option = ["Optimal Risky", "Minimum Volatility"]
    available_dates = load_csv("TradeDays.csv")

    # download necessary dependencies
    DOWNLOAD_TEXT = st.markdown("### Loading...Please wait")
    for fileName in EXTERNAL_DEPENDENCIES.keys():
        download_file(fileName)
    DOWNLOAD_TEXT.empty()


    # render the global plotly plots: line chart of 2018 past stock prices
    traces = []
    for stockCode in stock_option:
        df = load_csv(stockCode + ".csv")
        df = df[(df["Date"] >= "2018") & (df["Date"] < "2019")]
        traces.append(go.Scatter(
            x = df["Date"],
            y = df["Adj Close"],
            mode = "lines",
            name = stockCode,
        ))
    layout = dict(title = "Historical Stock Prices(Adjusted Close) of Topmost Famous US Stocks During 2018",
                  xaxis = dict(title = "Dates"),
                  yaxis = dict(title = "Price in USD"))
    fig = go.Figure(data = traces, layout = layout)
    st.plotly_chart(fig, use_container_width = True)

    # render sidebar
    st.sidebar.subheader("Configure Your Portfolio:")
    
    input_session_state = False
    invoke_session_state = False

    stock_selection = st.sidebar.multiselect(
        "Select Stocks to Analysis",
        options = stock_option,
        default = stock_option[0:5],
        disabled = input_session_state
    )
    if (len(stock_selection)) == 0:
        st.sidebar.warning("Cannot Start Portfolio Management with No Available Stocks.")
        invoke_session_state = True
    elif (len(stock_selection)) == 1:
        st.sidebar.warning("Cannot Start Portfolio Management with Only 1 Available Stocks.")
        invoke_session_state = True
    else:
        invoke_session_state = False

    risk_selection = st.sidebar.radio(
        "Select Preference for Risk Tolerance",
        options = risk_option,
        disabled = input_session_state
    )

    datestart_selection = st.sidebar.date_input(
        "Select a Date to Start Portofolio Management",
        value = datetime.date(2021, 1, 8),
        min_value = datetime.date(2019, 1, 1),
        max_value = datetime.date(2021, 12, 31),
        disabled = input_session_state
    )

    if not date_ValidCheck(datestart_selection, available_dates):
        st.sidebar.info("Oops! Seems that the selected date is NOT a trade day. Please try another day.")
        invoke_session_state = True
    
    daterange_selection = st.sidebar.slider(
        "Select a Date Range(by Days) to Perform Portfolio Management",
        min_value = 7,
        value = 14,
        step = 7,
        max_value = 35
    )

    # invoke service
    if st.sidebar.button(
        "Click to Start Portfolio Management!",
        disabled = invoke_session_state):
        # ______________________________________________________________ #
        # 1. predict stock prices:
        df_predictions, dict_predictions = predict_stockPrice(stock_selection, datestart_selection, daterange_selection)
        prediction_candlestichart(df_predictions, stock_selection)

        # ______________________________________________________________ #
        # 2. analyze portfolio management:

        # 在这里调用代码，返回值就 xx, yy = blahblah()
        # 写好了叫我来画图


        expander_2 = st.expander("Portfolio Management Outcomes:")
        expander_2.write(
            """
            TBC. See your Portfolio Managements Here(Check if it's clearred after invokes)
            """
        )
                

    # sample usage: render the plotly plots: candlestick chart
    # st_candlestichart("GOOG")

def portfolio_management(stock_selection,
                         risk_selection,
                         datestart_selection,
                         daterange_selection):
    # _________________________________________________________________ #
    # analyzes data

    # _________________________________________________________________ #
    # render website elements

    text = st.markdown("The Invoked Function is Running for 15 seconds.")
    info1 = st.markdown(stock_selection)
    info2 = st.markdown(risk_selection)
    info3 = st.markdown(datestart_selection)
    info4 = st.markdown(daterange_selection)
    time.sleep(15)
    text.empty()
    info1.empty()
    info2.empty()
    info3.empty()
    info4.empty()
    return


# ______________________________________________________________________________________________________________________
# chart render support

def prediction_candlestichart(df_predictions, stock_selection):
    with st.expander("Predicted Stock Prices:"):
        for stockCode in stock_selection:
            df = df_predictions[stockCode]

            trace1 = go.Scatter(x = pd.to_datetime(df["Date"]),
                            y = df["Adj Close"],
                            mode = "lines+markers",
                            marker = dict(color = "rgba(245, 210, 40, 0.8)"),
                            name = "Actual Adj Close")

            trace2 = go.Scatter(x = pd.to_datetime(df["Date"]),
                            y = df["Predicted_Adj_Close"],
                            mode = "lines+markers",
                            marker = dict(color = "rgba(40, 115, 240, 0.8)"),
                            name = "Predicted Adj Close")

            trace3 = go.Candlestick(
                    x = pd.to_datetime(df["Date"]),
                    open = df["Open"],
                    high = df["High"],
                    low = df["Low"],
                    close = df["Close"],
                    increasing_line_color = "green",
                    decreasing_line_color = "red",
                    name = "Stock Price",
                ),

            fig = go.Figure(
                data = trace3,
                layout = dict(
                title = "{} Stock Price in Candlestick Chart".format(stockCode),
                xaxis = dict(title = "Date"),
                yaxis = dict(title = "Price in USD")
                )
            )

            fig.add_trace(trace1)
            fig.add_trace(trace2)

            st.plotly_chart(fig, use_container_width = True)

    return


# ______________________________________________________________________________________________________________________
# validity check supports
def date_ValidCheck(datestart_selection: datetime.date,
                    df: pd.DataFrame):
    datestart = datestart_selection.strftime(r"%Y-%m-%d")
    df = load_csv("AAPL.csv")
    index = df[df["Date"].isin([datestart])].index

    if np.size(index) == 0:
        return False
    else:
        return True

# ______________________________________________________________________________________________________________________
# file resource supports

def load_csv(fileName):
    """
    Load csv files into pandas dataframe.
    """
    if os.path.exists((EXTERNAL_DEPENDENCIES[fileName]["directory"])):
        try:
            csv_path = EXTERNAL_DEPENDENCIES[fileName]["directory"] + "/" + fileName
            df = pd.read_csv(csv_path)
        except:
            raise Exception("File {} Load Fail!".format(fileName))

        return df
    else:
        raise Exception("File {} Not Exist!".format(fileName))


@st.cache
def download_file(fileName):
    """
    Download necessary dependencies with visualization in progress bar.
    """
    # initialize visual components to animate
    weights_warning = None
    progress_bar = None

    # set animation
    try:
        dst_folder = os.path.exists(EXTERNAL_DEPENDENCIES[fileName]["directory"])
        if not dst_folder:
            os.makedirs(EXTERNAL_DEPENDENCIES[fileName]["directory"])

        dst_path = EXTERNAL_DEPENDENCIES[fileName]["directory"] + "/" + fileName

        if os.path.exists(dst_path):
            # skip downloading if dependencies already exists
            return

        weights_warning = st.warning("Downloading %s..." % fileName)
        progress_bar = st.progress(0)

        with open(dst_path, "wb") as output:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[fileName]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0             # 2 ^ 20
                
                while True:
                    data = response.read(8192)      # Max Byte-Array Buffer Read at one time
                    if not data:
                        break                       # Load Compelete
                    counter += len(data)
                    output.write(data)

                    # operate animation by overwriting components
                    weights_warning.warning("Downloading %s...(%6.2f/%6.2f MB)" % 
                        (fileName, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # clear all components after downloading
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        
        if progress_bar is not None:
            progress_bar.empty()

    return


@st.experimental_singleton(show_spinner=False)      # experimental feature of function decorator to store singleton objects.
def load_file_content_as_string(path):
    """
    Load file content as strings.
    """
    repo_url = "https://raw.githubusercontent.com/Ch-i-Yu/Simple-Face-Mask-Detector/main" + "/" + path
    response = urllib.request.urlopen(repo_url)
    return response.read().decode("utf-8")

# ______________________________________________________________________________________________________________________
# analysis support
def predict_stockPrice(stock_selection: str,
                       datestart_selection: datetime.date,
                       daterange_selection: int):
    # initialize return values
    df_predictions = {}
    dict_predictions = {}
    
    for stockCode in stock_selection:
        df = load_csv(stockCode + ".csv")
        del df["Volume"]
        gc.collect()
        model = models.load_model(EXTERNAL_DEPENDENCIES[stockCode + ".h5"]["directory"] + r"/" + stockCode + ".h5")

        datestart = datestart_selection.strftime(r"%Y-%m-%d")

        date_index_base = df[df["Date"].isin([datestart])].index.values[0]
        df_base = df[(date_index_base - (daterange_selection - 1)) : (date_index_base + 1)]
        list_predictions = []

        for i in range(daterange_selection):
            index_prev = date_index_base - (daterange_selection - 1) + i
            index_post = date_index_base + 1 + i
            timestamp = np.array(df["Adj Close"][index_prev : index_post]).reshape(daterange_selection, 1)

            # Apply Feature Scaling to Test Data
            scaler = MinMaxScaler(feature_range = (0, 1))
            timestamp = np.reshape(timestamp, (-1, 1))
            timestamp = scaler.fit_transform(timestamp)
            timestamp = np.reshape(timestamp, (-1, daterange_selection))
            
            # Reshape Outputs
            pred = model.predict(timestamp)
            pred = scaler.inverse_transform(pred).tolist()
            list_predictions.append(pred[0][0])

        dict_predictions[stockCode] = list_predictions
        df_predictions[stockCode] = df_base.assign(Predicted_Adj_Close = list_predictions)

    return df_predictions, dict_predictions


# ______________________________________________________________________________________________________________________
# file resource external files to download
EXTERNAL_DEPENDENCIES = {
    # Stock Datasets
    "AAPL.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/AAPL.csv",
        "directory": "Stock-Datasets"
    },
    "AMZN.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/AMZN.csv",
        "directory": "Stock-Datasets"
    },
    "BRK-B.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/BRK-B.csv",
        "directory": "Stock-Datasets"
    },
    "GOOG.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/GOOG.csv",
        "directory": "Stock-Datasets"
    },
    "JNJ.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/JNJ.csv",
        "directory": "Stock-Datasets"
    },
    "JPM.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/JPM.csv",
        "directory": "Stock-Datasets"
    },
    "MSFT.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/MSFT.csv",
        "directory": "Stock-Datasets"
    },
    "NVDA.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/NVDA.csv",
        "directory": "Stock-Datasets"
    },
    "PG.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/PG.csv",
        "directory": "Stock-Datasets"
    },
    "TSLA.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/TSLA.csv",
        "directory": "Stock-Datasets"
    },
    "V.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/V.csv",
        "directory": "Stock-Datasets"
    },
    "WMT.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/WMT.csv",
        "directory": "Stock-Datasets"
    },
    "XOM.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Stock-Datasets/XOM.csv",
        "directory": "Stock-Datasets"
    },

    # Models
    "AAPL.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/AAPL.h5",
        "directory": "Models"
    },
    "AMZN.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/AMZN.h5",
        "directory": "Models"
    },
    "BRK-B.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/BRK-B.h5",
        "directory": "Models"
    },
    "GOOG.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/GOOG.h5",
        "directory": "Models"
    },
    "JNJ.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/JNJ.h5",
        "directory": "Models"
    },
    "JPM.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/JPM.h5",
        "directory": "Models"
    },
    "MSFT.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/MSFT.h5",
        "directory": "Models"
    },
    "NVDA.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/NVDA.h5",
        "directory": "Models"
    },
    "PG.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/PG.h5",
        "directory": "Models"
    },
    "TSLA.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/TSLA.h5",
        "directory": "Models"
    },
    "V.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/V.h5",
        "directory": "Models"
    },
    "WMT.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/WMT.h5",
        "directory": "Models"
    },
    "XOM.h5": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/Models/XOM.h5",
        "directory": "Models"
    },

    # Helper Resources
    "TradeDays.csv": {
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Portfolio-Management/main/TradeDays.csv",
        "directory": "Helper-Resources"
    },
}


if __name__ == "__main__":
    main()