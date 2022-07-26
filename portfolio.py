# -*- coding: utf-8 -*-

# @Date:        2022/07/26
# @Author:      Ashley-Willkes
# @Last Edited: 2022/07/26 23:27


from datetime import datetime
import numpy as np
import pandas as pd
# from pandas_datareader import data
#import matplotlib.pyplot as plt



class PortfolioManagement:
    def __init__(self,equities, preference, start, period):
        self.equities = equities
        self.preference = preference
        self.start = start
        self.period = period

    # equities:股票名，至少两种 str
    # preference:用户风险偏好,只有两种 str
    # start:起始时间 str
    # period:持续时间 int

    # 返回值为一个长度为period的列表。列表中顺序记录了每一天的优化组合投资策略。
    # 列表中每个元素包含：期望的月均回报率return 变化率（风险）Volatility 以及用户输入的各个股票的投资权重

    # predicted输入格式：一个字典。键为AAPL GOOG等字符串 List的长度和period相同。
    def Optimize(self, predicted):
        datas = {}
        count_out = self.period
        max_sharpe = 0


        ret_portfolios = []  # 返回值
        for equity in self.equities:
            file_relative = "./Stock-Datasets/" + equity + ".csv"
            result_csv = pd.read_csv(file_relative)
            result_csv = result_csv.drop(["Open", "High", "Low", "Close", "Volume"], axis=1)
            datas[equity] = result_csv

        time_offset = 0
        while count_out > 0:
            is_first = True
            for equity in self.equities:
                if is_first:
                    df = datas[equity]
                    newdata = pd.DataFrame(columns=None);
                    count = 0
                    record = 0
                    for i in df["Date"]:
                        i = datetime.strptime(i, '%Y-%m-%d')
                        i = datetime.date(i)
                        if i == self.start:
                            break
                        elif i == "2019-01-02":
                            record = count
                            count += 1
                        else:
                            count += 1
                    # insert = df[record:count + time_offset + 1]

                    left = count + time_offset - 27
                    right = count + time_offset + 1
                    insert = df[left:right]
                    # insert = df[record:count + time_offset + 1]
                    df_new = newdata.append(insert)


                    # 把当天的预测值赋给df
                    df_new.loc[count + time_offset, "Adj Close"] = predicted[equity][time_offset]
                    df_new.rename(columns={"Adj Close": equity}, inplace=True)
                    is_first = False

                else:
                    df_copy = datas[equity]
                    newdata_copy = pd.DataFrame(columns=None);
                    count = 0
                    record = 0
                    for i in df_copy["Date"]:
                        i = datetime.strptime(i, '%Y-%m-%d')
                        i = datetime.date(i)
                        if i == self.start:
                            break
                        elif i == "2019-01-02":
                            record = count
                            count += 1
                        else:
                            count += 1
                    insert = df_copy[count + time_offset - 27:count + time_offset + 1]
                    # insert = df_copy[record:count + time_offset + 1]
                    df_new_copy = newdata_copy.append(insert)

                    # 把当天的预测值赋给df，然后删除预测列
                    df_new_copy.loc[count + time_offset, "Adj Close"] = predicted[equity][time_offset]
                    df_new_copy = df_new_copy.drop(["Date"], axis=1)
                    df_new_copy.rename(columns={"Adj Close": equity}, inplace=True)

                    # 将新的股票列添加进来
                    columns = df_new.columns.tolist()
                    columns.insert(-1, equity)
                    df_new = df_new.reindex(columns=columns)
                    df_new[equity] = df_new_copy

            # df_new = df_new.drop([0])
            symbols = []
            for equity in self.equities:
                symbols.append(equity)
            df_new[symbols] = df_new[symbols].astype(float)
            # df_new.rename(columns={0: 'Date'}, inplace=True)
            df_new['Date'] = pd.to_datetime(df_new['Date'])
            df_new.set_index(['Date'], inplace=True)  # 用Date作index横坐标

            # 计算协方差、相关系数矩阵、Annual Standard Deviation
            cov_matrix = df_new.pct_change().apply(lambda x: np.log(1 + x)).cov()
            corr_matrix = df_new.pct_change().apply(lambda x: np.log(1 + x)).corr()
            ann_sd = df_new.pct_change().apply(lambda x: np.log(1 + x)).std().apply(lambda x: x * np.sqrt(250))
            # ind_er = df_new.resample('M').last().pct_change().mean()    #由于时间可能较短，用月均回报率代替年均回报率
            ind_er = df_new.pct_change().mean()
            assets = pd.concat([ind_er, ann_sd],
                               axis=1)  # Creating a table for visualising returns and volatility of assets
            assets.columns = ['Returns', 'Volatility']

            # 求权重，蒙特卡洛抽样
            p_ret = []  # Define an empty array for portfolio returns
            p_vol = []  # Define an empty array for portfolio volatility
            p_weights = []  # Define an empty array for asset weights

            num_assets = len(df_new.columns)
            num_portfolios = 10000
            for portfolio in range(num_portfolios):
                weights = np.random.random(num_assets)
                weights = weights / np.sum(weights)
                p_weights.append(weights)
                returns = np.dot(weights, ind_er)  # Returns are the product of individual
                # expected returnsof asset and its weights
                p_ret.append(returns)
                var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
                sd = np.sqrt(var)
                ann_sd = sd * np.sqrt(250)
                p_vol.append(ann_sd)

            data = {'Returns': p_ret, 'Volatility': p_vol}
            for counter, symbol in enumerate(df_new.columns.tolist()):
                # print(counter, symbol)
                data[symbol + 'weight'] = [w[counter] for w in p_weights]
            portfolios = pd.DataFrame(data)  # 得到蒙特卡洛抽样的组合投资策略

            # Optional: Plot efficient frontier
            # portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True,
            #                         figsize=[10, 10])

            if self.preference == 'Minimun Volatility':
                rf = 0.0001  # risk factor
                min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
                sharpe_ratio = ((min_vol_port['Returns'] - rf) / min_vol_port['Volatility']) * 100
                # print(sharpe_ratio)
                if (sharpe_ratio > max_sharpe):
                    max_sharpe = sharpe_ratio
                ret_portfolios.append(min_vol_port)
            else:
                rf = 0.0001  # risk factor
                optimal_risky_port = portfolios.iloc[((portfolios['Returns'] - rf) / portfolios['Volatility']).idxmax()]
                sharpe_ratio = ((optimal_risky_port['Returns'] - rf) / optimal_risky_port['Volatility']) * 100
                # print(sharpe_ratio)
                if (sharpe_ratio > max_sharpe):
                    max_sharpe = sharpe_ratio
                ret_portfolios.append(optimal_risky_port)
            count_out -= 1
            time_offset += 1
            
            ans = []
            ans.append(ret_portfolios)
            ans.append(max_sharpe)

        return ans
