import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

COST_IN_BPS = 5 #0.05%

warnings.filterwarnings('ignore')
prices = pd.read_csv('data/all_data_3.csv')
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

# Find the 20 best pairs
def gra_find_pairs(normalized_prices: pd.DataFrame, num_pairs: int) -> list[tuple[str,str]]:
    tickers = list(normalized_prices.keys())
    all_pairs = []

    # Find pairs associated with SPY
    ticker1 = "SPY"
    for j in range(len(tickers)):
        ticker2 = tickers[j]
        if ticker2 == ticker1:
            continue
        distance = gray_relation_coefficient(normalized_prices[ticker1], normalized_prices[ticker2])
        all_pairs.append((distance, (ticker1, ticker2)))

    all_pairs.sort()
    # Pick highest pairs (i.e., last ones)
    pairs = [pair for distance, pair in all_pairs[-num_pairs:]]

    return pairs

# Store top 20 pairs
top_20_pairs = gra_find_pairs(normalized_prices, 20)

print(top_20_pairs)
# Unpack top 20 pairs to just a list of tickers in the top 20 pairs
top_20_tickers = [ticker for pair in top_20_pairs for ticker in pair]
# Filter out all non-unique values
top_20_tickers = list(set(top_20_tickers))

print(top_20_tickers)

# Keep all columns of prices that are in the top 20 pairs
prices = prices[top_20_tickers]

# Define the rolling window size
window_size = 252  # For example, use one year of data (252 trading days)

# Initialize lists to store results
predicted_spy_prices = []
residuals_list = []
positions = []

# Loop over the price data with a rolling window
for end in range(window_size, len(prices)):
    start = end - window_size
    rolling_prices = prices.iloc[start:end]
    
    # Define the dependent and independent variables
    spy_prices = rolling_prices['SPY']
    other_etfs = rolling_prices.drop(columns=['SPY'])
    
    # Add a constant to the independent variables (for the intercept)
    X = sm.add_constant(other_etfs)
    y = spy_prices
    
    # Fit the linear regression model
    model = sm.OLS(y, X).fit()
    
    # Ensure the next_X DataFrame has the same structure as X
    next_X = prices[other_etfs.columns].iloc[end:end+1]
    next_X = sm.add_constant(next_X, has_constant='add')
    
    # Predict SPY price for the next day
    spy_pred = model.predict(next_X)
    predicted_spy_prices.append(spy_pred.values[0])
    
    # Calculate the residual (actual SPY price - predicted SPY price)
    actual_spy_price = prices['SPY'].iloc[end]
    residual = actual_spy_price - spy_pred.values[0]
    residuals_list.append(residual)
    
    # Determine the position based on the residual
    threshold = np.std(residuals_list[-window_size:])  # Calculate rolling threshold
    if residual > threshold:
        positions.append(1)  # Long SPY when residual is significantly positive (undervalued)
    elif residual < -threshold:
        positions.append(-1)  # Short SPY when residual is significantly negative (overvalued)
    else:
        positions.append(0)  # No position


# Align positions with the original index
positions = pd.Series(positions, index=prices.index[window_size:])

# For display positions
nonzero_positions = positions[positions != 0]
print(nonzero_positions.head())

# Plot residuals and positions
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(prices.index[window_size:], residuals_list, label='Residuals')
plt.axhline(threshold, color='r', linestyle='--', label='Positive Threshold')
plt.axhline(-threshold, color='g', linestyle='--', label='Negative Threshold')
plt.title('Residuals from Rolling Linear Regression')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(positions, label='Positions', marker='o')
plt.title('Trading Positions based on Residuals')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate daily returns of SPY
spy_returns = prices['SPY'].pct_change().fillna(0)

# Align returns with the positions
aligned_returns = spy_returns[window_size:]

# Calculate strategy returns
strategy_returns = aligned_returns * positions.shift(1).fillna(0)  # Shift positions to align with returns

# Calculate cumulative returns
cumulative_spy_returns = (1 + aligned_returns).cumprod() - 1
cumulative_strategy_returns = (1 + strategy_returns).cumprod() - 1

# Add swap cost every time the position changes
swap_fees = (np.abs(positions - positions.shift(1).fillna(0)) * COST_IN_BPS * 10**(-6)).cumsum()
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
