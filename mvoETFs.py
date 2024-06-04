import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the data
asset2 = pd.read_csv('etfarb.csv')
asset3 = pd.read_csv('cumulative_strategy_returns.csv')

# Convert the 'Date_' column to datetime to ensure proper merging
asset2['Date_'] = pd.to_datetime(asset2['Date_'])
asset3['Date_'] = pd.to_datetime(asset3['Date_'])
# Merge the dataframes on 'Date_'
data_merged = pd.merge(asset2, asset3, on='Date_', suffixes=('_asset2', '_asset3'))

# Print the column names to check if they are correct
print(data_merged.columns)

# Calculate mean returns and covariance matrix
# Update the column names based on the output from the previous step
mean_returns = data_merged[['Total_returns_asset2', 'Total_returns_asset3']].mean()
cov_matrix = data_merged[['Total_returns_asset2', 'Total_returns_asset3']].cov()

# Number of assets
num_assets = 2

# Rest of the code remains the same...
# Number of assets
num_assets = 2

# Define the objective function (portfolio variance)
def portfolio_variance(weights):
    return weights.T @ cov_matrix @ weights

# Constraints: weights sum to 1 and all weights are non-negative
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Sum of weights is 1
bounds = tuple((0, 1) for asset in range(num_assets))  # Weights between 0 and 1

# Initial guess
initial_weights = np.array([1/num_assets] * num_assets)

# Optimize
opt_results = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights
optimal_weights = opt_results.x

# Print the optimal weights
print("Optimal Weights:", optimal_weights)

# Calculate expected return and expected volatility of the optimal portfolio
expected_return = np.sum(mean_returns * optimal_weights)
expected_volatility = np.sqrt(opt_results.fun)

print("Expected Portfolio Return:", expected_return)
print("Expected Portfolio Volatility:", expected_volatility)


# Function to calculate portfolio performance for given weights
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std_dev, returns

# Generate random portfolios
num_portfolios = 10000
all_weights = np.zeros((num_portfolios, num_assets))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    all_weights[i, :] = weights
    portfolio_stats = portfolio_performance(weights, mean_returns, cov_matrix)
    vol_arr[i] = portfolio_stats[0]
    ret_arr[i] = portfolio_stats[1]

# Plotting the risk vs return of random portfolios
plt.figure(figsize=(10, 6))
plt.scatter(vol_arr, ret_arr, c='skyblue', label='Random Portfolios')
plt.scatter(expected_volatility, expected_return, c='red', s=50, label='Optimal Portfolio')
plt.title('Portfolio Optimization: Expected Return vs Volatility')
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the distribution of optimal weights
plt.figure(figsize=(8, 5))
plt.bar(range(len(optimal_weights)), optimal_weights, tick_label=['Naive ETFs', 'ETF over SPY'])
plt.title('Optimal Portfolio Weights Distribution')
plt.xlabel('Assets')
plt.ylabel('Weights')
plt.show()

# Calculate daily returns of the optimal portfolio
daily_returns = data_merged[['Total_returns_asset2', 'Total_returns_asset3']].dot(optimal_weights)

# Calculate cumulative returns
cumulative_returns = (1 + daily_returns).cumprod()

# Plotting the cumulative returns over time
plt.figure(figsize=(12, 6))
cumulative_returns.plot()
plt.title('Cumulative Returns of Optimal Portfolio Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.show()
