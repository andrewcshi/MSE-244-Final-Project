#script to use gray's relationship analysis on etf prices
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


COST_IN_BPS = 5 #0.05%
DAILY_EXPENSE_RATIO_BPS = 10/365 #0.10% annually


warnings.filterwarnings('ignore')
prices = pd.read_csv('data/all_data.csv')
warnings.filterwarnings('ignore')

metadata = pd.DataFrame(prices.iloc[:2])
metadata = metadata.T
metadata.columns = metadata.iloc[0]
metadata = metadata.iloc[1:]

prices = prices.iloc[2:]
prices.rename(columns={'Ticker':'Date'}, inplace=True)
prices['Date'] = pd.to_datetime(prices['Date'])
prices.set_index('Date', inplace=True, drop=True)
prices = prices.astype(float)

with pd.option_context('display.max_columns', 5):
    print(prices.head())
    print(metadata.head())

# Filter the data within years 2010 to 2023
start_date = '2010-01-04'
end_date = '2023-03-31'
prices = prices.loc[start_date:end_date]

# Remove tickers with insufficient data
prices = prices.dropna(axis=1, how='any')

# Calculate normalized prices and returns
normalized_prices = prices.div(prices.iloc[0])
returns = prices.pct_change().fillna(0)


def gray_relation_coefficient(reference, comparison, rho=0.5):
    diff = np.abs(reference - comparison)
    min_diff = np.min(diff)
    max_diff = np.max(diff)
    return np.mean((min_diff + rho * max_diff) / (diff + rho * max_diff))


#GRA finds how much association there is between a reference and a comparison set. a higher GRA means that more
#information is conveyed by the comparison set.
#find the 20 best pairs 
def gra_find_pairs(normalized_prices: pd.DataFrame, num_pairs: int) -> list[tuple[str,str]]:
    tickers = list(normalized_prices.keys())
    all_pairs = []

    #find pairs associated with spy
    ticker1 = "SPY"
    for j in range(len(tickers)):
        ticker2 = tickers[j]
        if ticker2 == ticker1:
            continue
        distance = gray_relation_coefficient(normalized_prices[ticker1], normalized_prices[ticker2])
        all_pairs.append((distance, (ticker1, ticker2)))

    all_pairs.sort()
    #because its gra, pick highest pairs (i.e. last ones)
    pairs = [pair for distance, pair in all_pairs[-num_pairs:]]

    return pairs

# Store top 20 pairs
top_20_pairs = gra_find_pairs(normalized_prices, 20)

print(top_20_pairs)
#unpack top 20 pairs to just a list of tickers in the top 20 pairs
top_20_tickers = [ticker for pair in top_20_pairs for ticker in pair]
#filter out all non-unique values
top_20_tickers = list(set(top_20_tickers))

print(top_20_tickers)

#kepp all columns of prices that are in the top 20 pairs
#that doesnt work try something else
prices = prices[top_20_tickers]


# Define the dependent and independent variables
spy_prices = prices['SPY']
other_etfs = prices.drop(columns=['SPY'])

# Add a constant to the independent variables (for the intercept)
X = sm.add_constant(other_etfs)
y = spy_prices

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Generate predictions for SPY prices
spy_pred = model.predict(X)

# Calculate residuals (actual SPY price - predicted SPY price)
residuals = spy_prices - spy_pred

# Determine positions based on residuals
threshold = residuals.std()  # You can adjust this threshold based on your strategy
positions = pd.Series(0, index=spy_prices.index)

positions[residuals > threshold] = 1  # Long SPY when residual is significantly positive (undervalued)
positions[residuals < -threshold] = -1  # Short SPY when residual is significantly negative (overvalued)

# For display positions
nonzero_positions = positions[positions != 0]
print(nonzero_positions.head())


# Plot residuals and positions
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(residuals, label='Residuals')
plt.axhline(threshold, color='r', linestyle='--', label='Positive Threshold')
plt.axhline(-threshold, color='g', linestyle='--', label='Negative Threshold')
plt.title('Residuals from Linear Regression')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(positions, label='Positions', marker='o')
plt.title('Trading Positions based on Residuals')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate daily price of SPY
spy_returns = spy_prices.pct_change().fillna(0)

# Calculate strategy returns
strategy_returns = spy_returns * positions.shift(1).fillna(0)  # Shift positions to align with returns


# Calculate cumulative returns
cumulative_spy_returns = (1 + spy_returns).cumprod() - 1
cumulative_strategy_returns = (1 + strategy_returns).cumprod() - 1

#add swap cost everytime the position changes
swap_fees = (np.abs(positions - positions.shift(1).fillna(0)) * COST_IN_BPS * 10**(-4)).cumsum()
cumulative_strategy_returns -= swap_fees

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(cumulative_spy_returns, label='Cumulative SPY Returns')
plt.plot(cumulative_strategy_returns, label='Cumulative Strategy Returns')
plt.title('Cumulative Returns of SPY and the Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

cumulative_strategy_returns.to_csv('cumulative_strategy_returns.csv')