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
import urllib
import time
import datetime

import pandas as pd
import numpy as np

from keras import models

from plotly import graph_objs as go
from plotly.figure_factory import create_candlestick
from plotly.graph_objects import Line, Marker



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

    # config global variables
    stock_option = ["AAPL", "AMZN", "BRK-B", "GOOG", "JNJ", "JPM",
                    "MSFT", "NVDA", "PG", "TSLA", "V", "WMT", "XOM"]
    risk_option = ["Optimal Risky", "Minimum Volatility"]

    # download necessary dependencies
    DOWNLOAD_TEXT = st.markdown("### Loading...Please wait")
    for fileName in EXTERNAL_DEPENDENCIES.keys():
        download_file(fileName)
    DOWNLOAD_TEXT.empty()

    # render the badges
    st.markdown(
        f"""
        [gh]: https://github.com/epogrebnyak/ssg-dataset
        [![GitHub Repo stars](https://img.shields.io/github/stars/epogrebnyak/ssg-dataset?style=social)][gh]
        """
    )

    # render the welcome text
    st.markdown("""
        ## Welcome to use Ka-Ching!
        TBC. *Hot Work in Progress!*
    """)

    # render the user guide
    with st.expander("User Guide:"):
        st.write("""
            1. Config your portfolio in the **`sidebar`**;
            2. Press the **`button`** below in the **`sidebar`** to start;
            
            Please double check the validity of your portfolio configurations if the **`button`** is disabled.
        """)

    # render the plotly plots: line chart
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

    # sample usage: render the plotly plots: candlestick chart
    st_candlestichart("GOOG")

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
        value = datetime.date(2020, 1, 1),
        min_value = datetime.date(2019, 1, 1),
        max_value = datetime.date(2021, 12, 31),
        disabled = input_session_state
    )
    
    daterange_selection = st.sidebar.slider(
        "Select a Date Range(by Days) to Perform Portfolio Management",
        min_value = 7,
        value = 14,
        step = 7,
        max_value = 35
    )

    if st.sidebar.button(
        "Click to Start Portfolio Management!",
         disabled = invoke_session_state):
         portfolio_management(stock_selection,
                              risk_selection,
                              datestart_selection,
                              daterange_selection)

def portfolio_management(stock_selection,
                         risk_selection,
                         datestart_selection,
                         daterange_selection):
    # initialize resources
    stockList = {}
    modelList = {}
    for stockCode in stock_selection:
        stockList[stockCode] = load_csv(stockCode + ".csv")
        modelList[stockCode] = models.load_model("Models" + "/" + stockCode + ".h5")

    # _________________________________________________________________ #
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

def st_candlestichart(stockCode):
    df = load_csv(stockCode + ".csv")
    df = df[(df["Date"] >= "2018") & (df["Date"] < "2019")]
    fig = go.Figure(
        data = go.Candlestick(
            x = pd.to_datetime(df["Date"]),
            open = df["Open"],
            high = df["High"],
            low = df["Low"],
            close = df["Close"],
            increasing_line_color = "green",
            decreasing_line_color = "red"
        ),
        layout = dict(
            title = "{} Stock Price in Candlestick Chart".format(stockCode),
            xaxis = dict(title = "Date"),
            yaxis = dict(title = "Price in USD")
        )
    )
    with st.expander("{} Stock Prices:".format(stockCode)):
        st.plotly_chart(fig, use_container_width = True)

    return

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

}


if __name__ == "__main__":
    main()