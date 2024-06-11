import pandas as pd
import numpy as np
import cvxpy as cp

# Load the data
asset1 = pd.read_csv('sorted_commodities_returns.csv')
asset2 = pd.read_csv('etfarb.csv')
asset3 = pd.read_csv('cumulative_strategy_returns.csv')

# Convert the 'Date' column to datetime to ensure proper merging
asset1['Date'] = pd.to_datetime(asset1['Date'])
asset2['Date'] = pd.to_datetime(asset2['Date'])
asset3['Date'] = pd.to_datetime(asset3['Date'])

# Merge the dataframes on 'Date'
data_merged = pd.merge(pd.merge(asset1, asset2, on='Date', suffixes=('_asset1', '_asset2')), asset3, on='Date')

print(data_merged.head())

data_merged.rename(columns={'Total_returns': 'Total_returns_asset3'}, inplace=True)

# Print the merged DataFrame to ensure it looks correct
print(data_merged.head())

from scipy.optimize import minimize

# Calculate mean returns and covariance matrix
mu = (data_merged[['Total_returns_asset1', 'Total_returns_asset2', 'Total_returns_asset3']].mean()).to_numpy()
Sigma = data_merged[['Total_returns_asset1', 'Total_returns_asset2', 'Total_returns_asset3']].cov().to_numpy()
print('--------')
print(type(mu))
print(Sigma.shape)
# Long only portfolio optimization.

n = 3

w = cp.Variable(n)

gamma = cp.Parameter(nonneg=True)
ret = mu.T @ w
risk = cp.quad_form(w, Sigma)
prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, w >= 0])
print(prob)
SAMPLES = 50
risk_data = np.zeros(SAMPLES)
ret_data = np.zeros(SAMPLES)
gamma_vals = np.logspace(-2, 3, num=SAMPLES)
for i in range(SAMPLES):
    gamma.value = gamma_vals[i]
    prob.solve()
    risk_data[i] = cp.sqrt(risk).value
    ret_data[i] = ret.value
print('---------------')
print(w.value)
# Plot long only trade-off curve.
import matplotlib.pyplot as plt



markers_on = [29, 40]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(risk_data, ret_data, "g-")
for marker in markers_on:
    plt.plot(risk_data[marker], ret_data[marker], "bs")
    ax.annotate(
        r"$\gamma = %.2f$" % gamma_vals[marker],
        xy=(risk_data[marker] + 0.08, ret_data[marker] - 0.03),
    )
    print(gamma_vals[marker])
# for i in range(n):
#     plt.plot(cp.sqrt(Sigma[i, i]).value, mu[i], "ro")
plt.xlabel("Standard deviation")
plt.ylabel("Return")
plt.show()

import scipy.stats as spstats

