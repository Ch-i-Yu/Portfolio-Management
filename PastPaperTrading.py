

# portfolios: 元素为dataframe的列表，dataframe中包含期望的月均回报率return 变化率（风险）Volatility 以及用户输入的各个股票的投资权重
# equities:股票名，至少两种 str
# start:起始时间 str
# period:持续时间 int
# profits:输出。为一个列表，列表内元素为浮点数，代表自start开始持续period个交易日的每日盈利（根据Portfolio的策略购买股票 本金为1w＄ 持股时间为一天）

def pastPaperTrading(period, start, equities, portfolios):
    count_loop = period
    datas = {}
    profits = []

    for equity in equities:
        file_relative = equity+".csv"
        result_csv = load_csv(file_relative)
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
                for i in df[0]:
                    if i == start:
                        break
                    else:
                        count += 1
                destination.append(count)
                insert = df[0:count + time_offset + 2]
                df_new = newdata.append(insert)

                # 删除预测列
                df_new = df_new.drop([2], axis=1)
                df_new.rename(columns={1: equity}, inplace=True)
                is_first = False

            else:
                df_copy = datas[equity]
                newdata_copy = pd.DataFrame(columns=None);
                count = 0
                for i in df_copy[0]:
                    if i == start:
                        break
                    else:
                        count += 1
                destination.append(count)
                insert = df_copy[0:count + time_offset + 2]
                df_new_copy = newdata_copy.append(insert)

                # 删除预测列
                df_new_copy = df_new_copy.drop([2], axis=1)
                df_new_copy = df_new_copy.drop([0], axis=1)
                df_new_copy.rename(columns={1: equity}, inplace=True)

                # 将新的股票列添加进来
                columns = df_new.columns.tolist()
                columns.insert(-1, equity)
                df_new = df_new.reindex(columns=columns)
                df_new[equity] = df_new_copy

        today_price = {}
        tomorrow_price = {}
        destination_count = 0

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
            profit += priciple * portfolios[time_offset][i] * (float(tomorrow_price[equity]) - float(today_price[equity]))

        profits.append(profit)

        time_offset += 1
        count_loop -= 1

    return profits





