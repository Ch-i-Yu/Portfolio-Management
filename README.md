# Portfolio-Management

A Portfolio Management Project Containing `Stock Price Predictio`n, `Portfolio Management`, `Past Paper Trading` and `Web App Visualization`.

---

## Package Designs are listed as below.

### Stock-Price-Prediction

```Python
Class StockPricePrediction:
    # Class Initialization.
    def _init_():
        self.args = None

    def _init_(arg):
        self.args = args

```
Example Pipelines for `StockPricePrediction:`
- Set Parameters(e.g. stockCode, EPOCHS, LOOKBACKS)
- Load Specified Single Stock Data
- Feature Engineering(Expanded with Percent Changes, Detect NaNs)
- Create Train/Test Datasets
- Build Model
  Plenty of Choices:
  XGBoost, your backup plan as always;
  Recurrent Neural Networks with LSTM, Highly recommended;
- Compile & Fit
- Plot Evaluations into PNG (We'll later use that for the Post)

### Portfolio Management
```Python
Class PortfolioManagement:
    # Class Initialization.
    def _init_():
        self.args = None

    def _init_(arg):
        self.args = args

```
Example Pipelines for `PortfolioManagement:`
- Utils functions to calculate Variance, Volatility, Covariance & Correlation Matrix, Expected Returns... All items mentioned in this [tutorial](https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/#optimal-risky-portfolio);
- Given "Look-Back" amount data, predict on future stocks;
- Building an Portfolio as mentioned in Step No.7;
- Form Covariance & Correlation Matrix based on a handful of historical data & the prediction based on which;
- Calculate Portfolio Variance, Portfolio Expected Returns
- Plotting the efficient frontier
(With MVO the Mean Variance Optimization. See also in this [website](https://www.effisols.com/basics/))


### Past Paper Trading
```Python
Class PastPaperTrading:
    # Class Initialization.
    def _init_():
        self.args = None

    def _init_(arg):
        self.args = args

```

Example Pipelines for `PastPaperTrading:`
- Input a TimePeriod(e.g. July 2018)
- Input a LookBack Period (default 14, 2 weeks)
- Input a handful of stocks
- Load Pre-Trained Models for Stocks Prediction
- Load LookBack Days + TimePeriod
- for 1st day of each week predict tomorrow's stock values then use it to decide an portfolio
- Apply Portfolio, Calculate Profits, Save Data to a Dataframe;
- Plot them after the period ends.

### Stock-Price-Prediction: A User's Walkthrough
1. A User Log on the Website
2. Ask User to input hyperparameters, e.g. start trading at which month, how long the look back is, aggressive or cautious regarding to the portfolio allocation
3. Ask User to input selected stocks from a stock pool.
4. Downloading External Files(stocks.csvs, models)
5. Use Progress Bar while performing portfolio allocation.
6. Plot a lot of metrics, the trading progress and the final outcome.