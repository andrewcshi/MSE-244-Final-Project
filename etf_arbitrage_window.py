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

# Use GGR distance formulation from class - subject to change
def ggr_distance(y: pd.Series, x: pd.Series) -> float:
    y_series = y.values
    x_series = x.values

    distance = np.sum((y_series - x_series) ** 2)
    return distance

# Discover top K pairs
def find_pairs(normalized_prices: pd.DataFrame, num_pairs: int) -> list[tuple[str,str]]:
    tickers = list(normalized_prices.keys())
    all_pairs = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            ticker1 = tickers[i]
            ticker2 = tickers[j]
            distance = ggr_distance(normalized_prices[ticker1], normalized_prices[ticker2])
            all_pairs.append((distance, (ticker1, ticker2)))

    all_pairs.sort()
    pairs = [pair for distance, pair in all_pairs[:num_pairs]]

    return pairs

# Store top 20 pairs
top_20_pairs = find_pairs(normalized_prices, 20)

# Estimate spreads between top K pairs
def estimate_spreads(pairs: list[tuple[str, str]], prices: pd.DataFrame, window_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    spread_dict = {}
    hedge_ratio_dict = {}

    for (ticker1, ticker2) in pairs:
        spread_series = []
        hedge_ratio_series = []

        for end in range(window_size, len(prices)):
            start = end - window_size
            y = prices[ticker1].iloc[start:end]
            x = prices[ticker2].iloc[start:end]
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            hedge_ratio = model.params[1]  # The slope of the regression

            spread = prices[ticker1].iloc[end] - hedge_ratio * prices[ticker2].iloc[end]
            spread_series.append(spread)
            hedge_ratio_series.append(hedge_ratio)

        spread_dict[f'{ticker1}-{ticker2}'] = pd.Series(spread_series, index=prices.index[window_size:])
        hedge_ratio_dict[f'{ticker1}-{ticker2}'] = pd.Series(hedge_ratio_series, index=prices.index[window_size:])

    spreads = pd.DataFrame(spread_dict)
    hedge_ratios = pd.DataFrame(hedge_ratio_dict)

    return spreads, hedge_ratios

# Store spreads and pairs trading hedge ratios
window_size = 252
spreads, hedge_ratios = estimate_spreads(top_20_pairs, prices, window_size)

# Translate signals to portfolio positions
def sig2pos(spreads: pd.DataFrame, df: pd.DataFrame, hedge_ratios: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    positions = pd.DataFrame(index=spreads.index, columns=df.columns).fillna(0)

    for pair in spreads.columns:
        spread = spreads[pair]
        ticker1, ticker2 = pair.split('-')

        hedge_ratio = hedge_ratios[pair]
        print("the hedge ratio series looks like:", hedge_ratio)

        # Short or long first security based on threshold
        positions.loc[spread > threshold, ticker1] = -1
        positions.loc[spread < -threshold, ticker1] = 1

        # Perform pairs trading on cointegrated security
        #hedgre ratio here is a series
        positions[ticker2] = -positions[ticker1] * hedge_ratio

    return positions

# Store positions
positions = sig2pos(spreads, prices, hedge_ratios)

print("positions are")
print(positions.head())

# For display positions
nonzero_positions = positions.loc[:, (positions != 0).any(axis=0)]
print(nonzero_positions.head())

#calculate trading fees for position. Impose a transaction cost for each switch in position
tmp = nonzero_positions.copy()
tmp = tmp.shift(1)
tmp.iloc[0] = 0
swap_fees = (np.abs(nonzero_positions - tmp) * COST_IN_BPS * 10**(-4)).cumsum().sum(axis=1)
print("fees are", swap_fees.tail())

#also impose a holding fee (annual expense ratio)
total_pos = nonzero_positions.copy().sum(axis=1)
holding_fee = (total_pos * DAILY_EXPENSE_RATIO_BPS * 10**(-4)).cumsum()
print("holding", holding_fee.tail())

# Plot 1: Heatmap of Positions
plt.figure(figsize=(12, 8))
sns.heatmap(nonzero_positions.T, cmap='coolwarm', cbar=True)
plt.xlabel('')
plt.ylabel('ETFs')

ax = plt.gca()
ax.set_xticks([])
plt.show()

print("shape is")
print(swap_fees.shape, (returns * nonzero_positions.shift(1)).sum(axis=1).cumsum().shape)

# Plot 2: Cumulative Returns
cumulative_returns = (returns * nonzero_positions.shift(1)).sum(axis=1).cumsum() - swap_fees - holding_fee
plt.figure(figsize=(12, 8))
cumulative_returns.plot()
plt.title('Cumulative Returns of the Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.show()
