# portfolios: 元素为dataframe的列表，dataframe中包含期望的月均回报率return 变化率（风险）Volatility 以及用户输入的各个股票的投资权重
# equities:股票名，至少两种 str
# start:起始时间 str
# period:持续时间 int
# profits:输出。为一个列表，列表内元素为浮点数，代表自start开始持续period个交易日的每日盈利（根据Portfolio的策略购买股票 本金为1w＄ 持股时间为一天）

def pastPaperTrading(period, start, equities, portfolios):
    count_loop = period
    datas = {}
    profits = []

    df_profit = pd.DataFrame(columns=['Date', 'Daily Profit'])
    df_forDrawing = pd.DataFrame(columns=['Date', 'Profit by Stock', 'Stock Code'])

    count_profit = 0
    count_drawing = 0

    for equity in equities:
        file_relative = "./Stock-Dataset/" + equity +".csv"
        result_csv = load_csv(file_relative)
        # result_csv = pd.read_csv(file_relative)
        result_csv = result_csv.drop(["Open", "High", "Low" , "Close" , "Volume"], axis=1)
        datas[equity] = result_csv

    time_offset = 0

    while count_loop > 0:
        is_first = True
        destination = []
        for equity in equities:
            if is_first:
                df = datas[equity]
                newdata = pd.DataFrame(columns=None);
                count = 0
                record = 0
                for i in df["Date"]:
                    if i == start:
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
                    if i == start:
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
        list = df_new.reset_index()['Date']
        date = list[destination[destination_count] - record + time_offset]
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

    return df_tuple





