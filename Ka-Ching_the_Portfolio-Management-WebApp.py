# -*- coding: utf-8 -*-

# @Date:        2022/07/25
# @Author:      Ch'i YU     Ashley Willkes
# @Last Edited: 2022/07/25 00:47
# @Project Contribution: TBC

"""
Some Text Here
The goal of this project is to ...
Anyway I'm going to name this website after "Ka-Ching!"
"""

# import necessary dependencies

import streamlit as st
import tensorflow
import os
import re
import gc
import urllib
import time
import datetime

from warnings import simplefilter
simplefilter(action = "ignore", category = FutureWarning)
simplefilter(action = "ignore", category = DeprecationWarning)

import pandas as pd
import numpy as np

from keras import models

from plotly import graph_objs as go
from plotly import express as px
from plotly.figure_factory import create_candlestick
from plotly.graph_objects import Line, Marker

from sklearn.preprocessing import MinMaxScaler

import portfolio

# ______________________________________________________________________________________________________________________
# main page

def main():
    """
    Starting `streamlit` execution in a main() function.
    """

    # set page title / layout / sidebar
    st.set_page_config(
        page_title="Ka-Ching: Visualize your AI-Supported Portfolio!",
        page_icon=":bar_chart:",
        initial_sidebar_state="expanded",
        layout="wide")

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
    tab_about, tab_overview, tab_userguide, = st.tabs(["About Us", "Overview", "User Guide"])
    with tab_about:
        st.markdown(
            """
            **What is `Ka-Ching`?**
            
            `Ka-Ching` is a slang word indicating the ringing of the bell of a cash register being opened, often as way of indicating profitability:money_with_wings:.
            
            **What does `Ka-Ching` do for me?**
            
            `Ka-Ching` uses AI(LSTM Network) to build your portfolios based on your preference and visualize them with interpretable `Plotly` plots.

            *Don't forget to star & fork our website if you like it*:sparkling_heart:

            """
        )
    with tab_userguide:
        st.markdown(
            """
            **For Plotly Charts**:
            - Double Click on legends to isolate a trace;
            - Single Click on legends to remove a trace;
            - Repeat your clicks to undo your operations;


            **For Our Portfolio Management:**
            1. Config your portfolio in the **`sidebar`**;
            2. Press the **`button`** below in the **`sidebar`** to start;

            Please double check the validity of your portfolio configurations if the **`button`** is disabled.
            """
        )
    with tab_overview:
        st.markdown(
            """
            **Stock Price Prediction**
            
            Based on formal data of stock market, LSTM (Long Short-Term Memory Network) with a lookback of 14 days could help predict future trends and thus yielding significant profit.
            """
        )
        st.markdown(
            """
            **MPT Portfolio Management**
            
            Ensure a flexibility in portfolios for you with  `Modern Portfolio Theory` aka `Mean-Variance Analysis` and implemented with `Monte Carlo Method`.
            """
        )
        st.markdown(
            """
            **Past Paper Trading**
            
            `Terrific Trading Simulation` on historical stock data with Full options to configure your portfolio(Stocks, Dates, Ranges and etc.)!
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
            x=df["Date"],
            y=df["Adj Close"],
            mode="lines",
            name=stockCode,
        ))
    layout = dict(title="Historical Stock Prices(Adjusted Close) of Topmost Famous US Stocks During 2018",
                  xaxis=dict(title="Dates"),
                  yaxis=dict(title="Price in USD"))
    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    # render sidebar
    st.sidebar.subheader("Configure Your Portfolio:")

    input_session_state = False
    invoke_session_state = False

    stock_selection = st.sidebar.multiselect(
        "Select Stocks to Analysis",
        options=stock_option,
        default=stock_option[0:3],
        disabled=input_session_state
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
        options=risk_option,
        disabled=input_session_state
    )

    datestart_selection = st.sidebar.date_input(
        "Select a Date to Start Portofolio Management",
        value=datetime.date(2021, 1, 8),
        min_value=datetime.date(2019, 1, 1),
        max_value=datetime.date(2021, 12, 31),
        disabled=input_session_state
    )

    if not date_ValidCheck(datestart_selection, available_dates):
        st.sidebar.info("Oops! Seems that the selected date is NOT a trade day. Please try another day.")
        invoke_session_state = True

    daterange_selection = st.sidebar.slider(
        "Select a Date Range(by Days) to Perform Portfolio Management",
        min_value = 7,
        value = 7,
        step = 7,
        max_value = 35
    )

    # invoke service
    if st.sidebar.button(
            "Click to Start Portfolio Management!",
            disabled=invoke_session_state):

        WAIT_INFO = st.info("Busying Analyzing...Please Wait for several minutes!")
        # ______________________________________________________________ #
        # 1. predict stock prices:
        df_predictions, dict_predictions = predict_stockPrice(stock_selection, datestart_selection, daterange_selection)
        prediction_candlestichart(df_predictions, stock_selection)

        # ______________________________________________________________ #
        # 2. analyze portfolio management:

        # Portfolio的返回值
        ans = portfolio_management(stock_selection=stock_selection , risk_selection= risk_selection,
                                  datestart_selection = datestart_selection,
                                  daterange_selection = daterange_selection,
                                  dict_predictions=dict_predictions)
        pf_ret = ans[0]
        sharpe_ratio = ans[1]

        df_csv = pd.DataFrame()
        for item in pf_ret:
            df_csv = df_csv.join(item)

        # PastPaperTrading的返回值
        df_profit, df_forDrawing = pastPaperTrading(period=daterange_selection,
                                   start=datestart_selection,
                                   equities=stock_selection,
                                   portfolios=pf_ret)

        portfolio_barchart(pf_ret, df_profit, df_forDrawing ,sharpe_ratio)

        # ______________________________________________________________ #
        # 3. complete and draw celebratory balloons
        st.balloons()
        WAIT_INFO.empty()
        st.download_button(label = "Download your portfolio in weights!",
                           data = df_csv.to_csv().encode("utf-8"),
                           mime = "text/csv" )


def portfolio_management(stock_selection,
                         risk_selection,
                         datestart_selection,
                         daterange_selection,
                         dict_predictions):
    # _________________________________________________________________ #
    # analyzes data
    pf = portfolio.PortfolioManagement(start = datestart_selection,
                                       equities = stock_selection,
                                       preference = risk_selection,
                                       period = daterange_selection)
    ans = pf.Optimize(predicted=dict_predictions)
    return ans


# portfolios: 元素为dataframe的列表，dataframe中包含期望的月均回报率return 变化率（风险）Volatility 以及用户输入的各个股票的投资权重
# equities:股票名，至少两种 str
# start:起始时间 str
# period:持续时间 int
# profits:输出。为一个列表，列表内元素为浮点数，代表自start开始持续period个交易日的每日盈利（根据Portfolio的策略购买股票 本金为1w＄ 持股时间为一天）

def pastPaperTrading(period, start, equities, portfolios):
    count_loop = period
    datas = {}
    start_now = start.strftime('%Y-%m-%d')
    profits = []

    df_profit = pd.DataFrame(columns=['Date', 'Daily Profit'])
    df_forDrawing = pd.DataFrame(columns=['Date', 'Profit by Stock', 'Stock Code'])

    count_profit = 0
    count_drawing = 0

    for equity in equities:
        # 有点小问题 忘记+s
        file_relative = equity + ".csv"
        result_csv = load_csv(file_relative)
        # result_csv = pd.read_csv(file_relative)
        result_csv = result_csv.drop(["Open", "High", "Low" , "Close" , "Volume"], axis=1)
        datas[equity] = result_csv

    time_offset = 0

    while count_loop > 0:
        # 打印循环次数
        # st.markdown("Loop: ")
        # st.markdown(count_loop)
        is_first = True
        destination = []
        for equity in equities:
            if is_first:
                df = datas[equity]
                newdata = pd.DataFrame(columns=None);
                count = 0
                record = 0
                for i in df["Date"]:
                    if i == start_now:
                        break
                    elif i == "2019-01-02":
                        record = count
                        count += 1
                    else:
                        count += 1
                destination.append(count)
                # +2是因为要计算利润 引入第二天的数据
                insert = df[record:count + time_offset + 2]
                df_new = newdata.append(insert)

                # 以股票名字重命名
                df_new.rename(columns={"Adj Close": equity}, inplace=True)
                is_first = False

            else:
                df_copy = datas[equity]
                newdata_copy = pd.DataFrame(columns=None);
                count = 0
                record = 0

                for i in df_copy["Date"]:
                    if i == start_now:
                        # st.markdown("Match!")
                        break
                    elif i == "2019-01-02":
                        record = count
                        count += 1
                    else:

                        count += 1
                destination.append(count)
                insert = df_copy[record:count + time_offset + 2]
                df_new_copy = newdata_copy.append(insert)

                # 删除预测列
                df_new_copy = df_new_copy.drop(["Date"], axis=1)
                df_new_copy.rename(columns={"Adj Close": equity}, inplace=True)

                # 将新的股票列添加进来
                columns = df_new.columns.tolist()
                columns.insert(-1, equity)
                df_new = df_new.reindex(columns=columns)
                df_new[equity] = df_new_copy

        today_price = {}
        tomorrow_price = {}
        destination_count = 0

        dates = []
        # 生成日期列表
        date_list = df_new.reset_index()['Date']
        time.sleep(15)
        # date = date_list[destination[destination_count] - record + time_offset]
        date = date_list[count - record + time_offset]

        dates.append(date)

        for equity in equities:

            today = df_new[equity][destination[destination_count]+time_offset]
            tomorrow = df_new[equity][destination[destination_count] +time_offset + 1]
            today_price[equity] = today
            tomorrow_price[equity] = tomorrow
            destination_count += 1

        profit = 0
        priciple = 10000

        for equity in equities:
            i = equity+"weight"
            profit_for_equity = priciple * portfolios[time_offset][i] * (float(tomorrow_price[equity]) - float(today_price[equity]))
            profit += profit_for_equity

            # 按日期 具体股票种类及该股票当日盈利为单位新建dataframe行
            df_forDrawing.loc[count_drawing] = [date, profit_for_equity, equity]
            count_drawing += 1

        df_profit.loc[count_profit] = [date, profit]
        count_profit += 1

        # profits.append(profit)
        df_tuple = (df_profit, df_forDrawing)

        time_offset += 1
        count_loop -= 1
    # 打印完成记录
    # st.markdown("Done!")
    return df_tuple

# ______________________________________________________________________________________________________________________
# chart render support

def prediction_candlestichart(df_predictions, stock_selection):
    with st.expander("Predicted Stock Prices:"):
        for stockCode in stock_selection:
            df = df_predictions[stockCode]

            trace1 = go.Scatter(x = pd.to_datetime(df["Date"]),
                                y = df["Adj Close"],
                                mode = "lines+markers",
                                marker = dict(color="rgba(245, 210, 40, 0.8)"),
                                name = "Actual Adj Close")

            trace2 = go.Scatter(x=pd.to_datetime(df["Date"]),
                                y=df["Predicted_Adj_Close"],
                                mode="lines+markers",
                                marker=dict(color="rgba(40, 115, 240, 0.8)"),
                                name="Predicted Adj Close")

            trace3 = go.Candlestick(
                x=pd.to_datetime(df["Date"]),
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color="green",
                decreasing_line_color="red",
                name="Stock Price",
            ),

            fig = go.Figure(
                data=trace3,
                layout=dict(
                    title="{} Stock Price in Candlestick Chart".format(stockCode),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Price in USD")
                )
            )

            fig.add_trace(trace1)
            fig.add_trace(trace2)

            st.plotly_chart(fig, use_container_width=True)

    return


def portfolio_barchart(pf_ret, df_profit, df_forDrawing, sharpe_ratio):
    with st.expander("Analyzed Portfolio Outcomes:"):
        col1, col2 ,col3= st.columns(3)
        sum = 0
        count = 0
        for i in pf_ret:
            sum += abs(i["Returns"])
            count += 1
#       col1.metric("Returns", pf_ret[-1]["Returns"])

        col1.metric("Returns", "{:.2%}".format(sum))
        col2.metric("Volatility", "{:.2%}".format(pf_ret[-1]["Volatility"]))
        col3.metric("Sharpe Ratio", "{:.2%}".format(sharpe_ratio))
        
        fig = px.bar(df_forDrawing,
                    x = "Date",
                    y = "Profit by Stock",
                    color = "Stock Code",
                    title = "Portfolio Outcomes")

        trace = go.Scatter(
            x = df_profit["Date"],
            y = df_profit["Daily Profit"],
            mode = "lines+markers",
            marker = dict(color="rgba(255, 255, 102, 0.8)"),
            name = "Daily Profit"
        )

        fig.add_trace(trace)
        st.plotly_chart(fig, use_container_width=True)

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
                MEGABYTES = 2.0 ** 20.0  # 2 ^ 20

                while True:
                    data = response.read(8192)  # Max Byte-Array Buffer Read at one time
                    if not data:
                        break  # Load Compelete
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


@st.experimental_singleton(show_spinner=False)  # experimental feature of function decorator to store singleton objects.
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
        df_outcome = df[(date_index_base) : (date_index_base + daterange_selection)]
        list_predictions = []

        for i in range(daterange_selection):
            index_prev = date_index_base - (14 - 1) + i
            index_post = date_index_base + 1 + i
            timestamp = np.array(df["Adj Close"][index_prev: index_post])

            # Apply Feature Scaling to Test Data
            scaler = MinMaxScaler(feature_range=(0, 1))
            timestamp = np.reshape(timestamp, (-1, 1))
            timestamp = scaler.fit_transform(timestamp)
            timestamp = np.reshape(timestamp, (-1, 14))

            # Reshape Outputs
            pred = model.predict(timestamp)
            pred = scaler.inverse_transform(pred).tolist()
            list_predictions.append(pred[0][0])

        dict_predictions[stockCode] = list_predictions
        df_predictions[stockCode] = df_outcome.assign(Predicted_Adj_Close = list_predictions)

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
