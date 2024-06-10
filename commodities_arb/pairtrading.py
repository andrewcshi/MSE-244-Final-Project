import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import drawdowns, cumulative_returns, annualized_sharpe_ratio, annualized_return, annualized_volatility, skewness, kurtosis, max_drawdown

# Load your data with dtype specification to avoid mixed type warnings
print('Loading data...')
dtype_spec = {
    'FutCode': str,
    'Date_': str,
    'Settlement': float,
}

# List of pairs
commodity_pairs = [
    ("commodities/CoffeeC.csv", "commodities/feedercattle.csv"),
    #("commodities/soybeanmeal.csv", "commodities/soybeancomposite.csv"),
    ("commodities/LeanHogsComposite.csv", "commodities/livecattlecomposite.csv"),
    ("commodities/soybreal-oil.csv", "commodities/wheatcomposite.csv"),
    ("commodities/HenryHubNaturalGas FinancialLastDayComposite.csv", "commodities/GasolineRBOB.csv"),
    #("commodities/BrentCrudeOilLastDay.csv", "commodities/MontBelvieuLDHPropane.csv")
]

# Function to load and clean data
def load_and_clean_data(file):
    df = pd.read_csv(file, dtype=dtype_spec, low_memory=False)
    df['Date_'] = pd.to_datetime(df['Date_'])
    df = df[['FutCode', 'Date_', 'Settlement']]
    df = df.sort_values(by=['FutCode', 'Date_'])
    return df

def find_matching_contracts(data1, data2):
    data1_contracts = data1['FutCode'].unique()
    data2_contracts = data2['FutCode'].unique()

    matching_pairs = []
    for contract1 in data1_contracts:
        # Filter dates for contract1 to only include those after 2012
        data1_dates = data1[(data1['FutCode'] == contract1) & (data1['Date_'] > '2010-12-31')]['Date_']
        for contract2 in data2_contracts:
            # Filter dates for contract2 to only include those after 2012
            data2_dates = data2[(data2['FutCode'] == contract2) & (data2['Date_'] > '2010-12-31')]['Date_']
            common_dates = data1_dates[data1_dates.isin(data2_dates)]
            if len(common_dates) > 0:
                matching_pairs.append((contract1, contract2, common_dates))
                if len(matching_pairs) == 5:
                    return matching_pairs
    return matching_pairs
# #Function to find 5 matching contracts over time
# def find_matching_contracts(data1, data2):
#     data1_contracts = data1['FutCode'].unique()
#     data2_contracts = data2['FutCode'].unique()
    
#     matching_pairs = []
#     for contract1 in data1_contracts:
#         data1_dates = data1[data1['FutCode'] == contract1]['Date_']
#         for contract2 in data2_contracts:
#             data2_dates = data2[data2['FutCode'] == contract2]['Date_']
#             common_dates = data1_dates[data1_dates.isin(data2_dates)]
#             if len(common_dates) > 0:
#                 matching_pairs.append((contract1, contract2, common_dates))
#                 if len(matching_pairs) == 5:
#                     return matching_pairs
#     return matching_pairs

# def process_data(data1, data2, contract1, contract2, common_dates):
#     data1_selected = data1[(data1['FutCode'] == contract1) & (data1['Date_'].isin(common_dates))]
#     data2_selected = data2[(data2['FutCode'] == contract2) & (data2['Date_'].isin(common_dates))]
    
#     # Ensure we only have data within the contract periods without extending them
#     data1_selected = data1_selected.set_index('Date_').reindex(common_dates).ffill().reset_index()
#     data2_selected = data2_selected.set_index('Date_').reindex(common_dates).ffill().reset_index()
    
#     data1_selected['Normalized'] = data1_selected['Settlement'] / data1_selected['Settlement'].iloc[0]
#     data2_selected['Normalized'] = data2_selected['Settlement'] / data2_selected['Settlement'].iloc[0]
    
#     merged_data = pd.merge(data1_selected[['Date_', 'Normalized']], data2_selected[['Date_', 'Normalized']], on='Date_', suffixes=('_1', '_2'))
#     return merged_data
def process_data(data1, data2, contract1, contract2, common_dates):
    data1_selected = data1[(data1['FutCode'] == contract1) & (data1['Date_'].isin(common_dates))]
    data2_selected = data2[(data2['FutCode'] == contract2) & (data2['Date_'].isin(common_dates))]
    
    # Ensure we only have data within the contract periods without extending them
    data1_selected = data1_selected.set_index('Date_').reindex(common_dates).ffill().reset_index()
    data2_selected = data2_selected.set_index('Date_').reindex(common_dates).ffill().reset_index()
    
    data1_selected['Normalized'] = data1_selected['Settlement'] / data1_selected['Settlement'].iloc[0]
    data2_selected['Normalized'] = data2_selected['Settlement'] / data2_selected['Settlement'].iloc[0]
    
    merged_data = pd.merge(data1_selected[['Date_', 'Normalized', 'Settlement']],
                           data2_selected[['Date_', 'Normalized', 'Settlement']],
                           on='Date_', suffixes=('_1', '_2'))
    return merged_data

# Initialize an empty DataFrame for portfolio returns
portfolio_returns = pd.DataFrame()

# Iterate over each pair and apply pairs trading strategy
for file1, file2 in commodity_pairs:
    data1 = load_and_clean_data(file1)
    data2 = load_and_clean_data(file2)
    
    matching_pairs = find_matching_contracts(data1, data2)
    
    if len(matching_pairs) < 5:
        print(f'Not enough matching contracts found for pair: {file1} and {file2}')
        continue
    
    for contract1, contract2, common_dates in matching_pairs:
        print(f'Selected matching pair: {contract1} and {contract2}')
        
        # Process data for the selected pair of contracts
        merged_data = process_data(data1, data2, contract1, contract2, common_dates)
        
        # Drop rows with any NaN or infinite values
        merged_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged_data.dropna(inplace=True)
        
        if merged_data.empty or merged_data['Normalized_1'].empty or merged_data['Normalized_2'].empty:
            print(f'Empty data after processing for pair: {contract1} and {contract2}')
            continue
        
        # Estimate spread and hedge ratio using linear regression
        y = merged_data['Normalized_1']
        x = merged_data['Normalized_2']
        x = sm.add_constant(x)
        
        if x.shape[0] == 0 or y.shape[0] == 0:
            print(f'Empty arrays for regression for pair: {contract1} and {contract2}')
            continue
        
        model = sm.OLS(y, x).fit()
        hedge_ratio = model.params[1]
        
        # Calculate the spread
        merged_data['Spread'] = y - hedge_ratio * merged_data['Normalized_2']
        
        # Define trading signals using z-score
        merged_data['Spread_mean'] = merged_data['Spread'].rolling(window=30).mean()
        merged_data['Spread_std'] = merged_data['Spread'].rolling(window=30).std()
        merged_data['Z-score'] = (merged_data['Spread'] - merged_data['Spread_mean']) / merged_data['Spread_std']
        
        # Handle NaNs in rolling calculations
        merged_data['Spread_mean'].fillna(method='ffill', inplace=True)
        merged_data['Spread_std'].fillna(method='ffill', inplace=True)
        merged_data['Z-score'].fillna(method='ffill', inplace=True)
        
        # Generate buy and sell signals
        entry_threshold = 2
        exit_threshold = 0
        merged_data['Signal'] = np.where(merged_data['Z-score'] > entry_threshold, -1, np.nan)  # Short signal
        merged_data['Signal'] = np.where(merged_data['Z-score'] < -entry_threshold, 1, merged_data['Signal'])  # Long signal
        merged_data['Signal'] = np.where(abs(merged_data['Z-score']) < exit_threshold, 0, merged_data['Signal'])  # Exit signal
        
        # Fill the NaN signals forward
        merged_data['Signal'] = merged_data['Signal'].ffill().fillna(0)
        
        # Translate signals to positions
        merged_data['Position_1'] = merged_data['Signal']
        merged_data['Position_2'] = -merged_data['Signal'] * hedge_ratio
        
        # Calculate strategy returns
        # merged_data['Return_1'] = merged_data['Position_1'].shift() * merged_data['Normalized_1'].pct_change()
        # merged_data['Return_2'] = merged_data['Position_2'].shift() * merged_data['Normalized_2'].pct_change()
        # merged_data['Strategy_returns'] = merged_data['Return_1'] + merged_data['Return_2']
                # Calculate strategy returns using actual settlement prices
        merged_data['Return_1'] = merged_data['Position_1'].shift() * (merged_data['Settlement_1'].pct_change())
        merged_data['Return_2'] = merged_data['Position_2'].shift() * (merged_data['Settlement_2'].pct_change())
        merged_data['Strategy_returns'] = merged_data['Return_1'] + merged_data['Return_2']

        
        # Append strategy returns to the portfolio
        #portfolio_returns['Cumulative_returns'] = cumulative_returns(portfolio_returns['Total_returns'])

        if portfolio_returns.empty:
            portfolio_returns = merged_data[['Date_', 'Strategy_returns']]
        else:
            portfolio_returns = pd.merge(portfolio_returns, merged_data[['Date_', 'Strategy_returns']], on='Date_', how='outer', suffixes=('', '_'+contract1+'_'+contract2))

# Fill missing values with 0
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define a function to calculate cumulative returns
def cumulative_returns(returns):
    return (1 + returns).cumprod() - 1

# Adjusted code for computing cumulative returns and plotting
portfolio_returns.fillna(0, inplace=True)

# Calculate total returns for each day
portfolio_returns['Total_returns'] = portfolio_returns.drop(columns=['Date_']).sum(axis=1)

# Calculate cumulative returns
portfolio_returns['Cumulative_returns'] = cumulative_returns(portfolio_returns['Total_returns'])

# Plot the cumulative returns
print('Plotting results...')
plt.figure(figsize=(10, 6))
portfolio_returns['Date_'] = pd.to_datetime(portfolio_returns['Date_'])  # Ensure Date_ is in datetime format
portfolio_returns.set_index('Date_', inplace=True)
portfolio_returns['Cumulative_returns'].plot(title='Cumulative Returns from Combined Pairs Trading Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.show()
print('Finished')

# Save the total returns to a CSV file
portfolio_returns.reset_index().to_csv('commodities_returns.csv', columns=['Date_', 'Total_returns'], index=False)

# Read the CSV file
data = pd.read_csv("commodities_returns.csv")

# Sort by the 'Date_' column in ascending order (change to 'descending' for newest to oldest)
sorted_data = data.sort_values(by='Date_', ascending=True)

sorted_data.to_csv("sorted_commodities_returns.csv", index=False)
# Calculate metrics
cumulative_ret = cumulative_returns(portfolio_returns['Total_returns'])
drawdown = drawdowns(portfolio_returns['Total_returns'])
sharpe_ratio = annualized_sharpe_ratio(portfolio_returns['Total_returns'])
ann_return = annualized_return(portfolio_returns['Total_returns'])
ann_volatility = annualized_volatility(portfolio_returns['Total_returns'])
port_skewness = skewness(portfolio_returns['Total_returns'])
port_kurtosis = kurtosis(portfolio_returns['Total_returns'])
max_ddown = max_drawdown(portfolio_returns['Total_returns'])

print('cumret', cumulative_ret)

print('drawdown', drawdown)

print('sharpe', sharpe_ratio)

print('ret', ann_return)
print('vol', ann_volatility)
print('skew', port_skewness)
print('kurt', port_kurtosis)
print('ddown', max_ddown)
