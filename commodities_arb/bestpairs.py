# import pandas as pd
# import numpy as np
# import os

# data_dir = "commodities"

# def load_and_clean_data(file):
#     df = pd.read_csv(os.path.join(data_dir, file))
#     df['Date_'] = pd.to_datetime(df['Date_'])
#     df = df.sort_values(by='Date_')
#     df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Settlement'])
#     df = df[df['Settlement'] != 0]
#     df['Normalized_Settlement'] = (df['Settlement'] - df['Settlement'].mean()) / df['Settlement'].std()
#     return df

# commodity_files = ["corn.csv", "wheatcomposite.csv"]
# data = {file.split('.')[0]: load_and_clean_data(file) for file in commodity_files}

# def filter_data(df):
#     return df[(df['Date_'] >= '2000-01-01') & (df['Date_'] <= '2010-12-31')]

# wheat_data = filter_data(data['wheatcomposite'])
# corn_data = filter_data(data['corn'])

# merged_data = pd.merge(wheat_data[['Date_', 'Normalized_Settlement']], 
#                        corn_data[['Date_', 'Normalized_Settlement']], 
#                        on='Date_', 
#                        suffixes=('_wheat', '_corn'))

# correlation = merged_data[['Normalized_Settlement_wheat', 'Normalized_Settlement_corn']].corr()
# print(correlation)
import pandas as pd
import numpy as np
import os
from arch.unitroot import engle_granger
from itertools import combinations, permutations

# Set the directory containing your CSV files
data_dir = "commodities"

# List of commodity CSV files
commodity_files = [
    "BrentCrudeOilLastDay.csv",
    "CoffeeC.csv",
    "corn.csv",
    "cotton2.csv",
    "crudeoil.csv",
    "feedercattle.csv",
    "GasolineRBOB.csv",
    "HeatingOil.csv",
    "HenryHubNaturalGas FinancialLastDayComposite.csv",
    "LeanHogsComposite.csv",
    "livecattlecomposite.csv",
    "lumber.csv",
    "NaturalGas.csv",
    "MontBelvieuLDHPropane.csv",
    "Sugar11.csv",
    "wheatcomposite.csv",
    "soybeancomposite.csv",
    "soybeanmeal.csv",
    "soybreal-oil.csv"
]

# Function to load, clean, and normalize data
def load_and_clean_data(file):
    df = pd.read_csv(os.path.join(data_dir, file))
    df['Date_'] = pd.to_datetime(df['Date_'])
    df = df.sort_values(by='Date_')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Settlement'])
    df = df[df['Settlement'] != 0]
    df['Normalized_Settlement'] = (df['Settlement'] - df['Settlement'].mean()) / df['Settlement'].std()
    return df

# Load data into a dictionary of DataFrames
data = {file.split('.')[0]: load_and_clean_data(file) for file in commodity_files}

# Function to filter data between 2000 and 2010
def filter_data(df):
    return df[(df['Date_'] >= '2000-01-01') & (df['Date_'] <= '2010-12-31')]

# Find best cointegrated pairs
results = []
pairs = list(permutations(data.keys(), 2))

for pair in pairs:
    df1 = filter_data(data[pair[0]])
    df2 = filter_data(data[pair[1]])
    
    merged_data = pd.merge(df1[['Date_', 'Normalized_Settlement']], 
                           df2[['Date_', 'Normalized_Settlement']], 
                           on='Date_', 
                           suffixes=('_' + pair[0], '_' + pair[1]))
    
    if not merged_data.empty:
        # Perform Engle-Granger test with optimized max_lags
        test_result = engle_granger(merged_data['Normalized_Settlement_' + pair[0]], 
                                    merged_data['Normalized_Settlement_' + pair[1]],
                                    max_lags=10)
        p_value = test_result.pvalue
        results.append((pair[0], pair[1], p_value))
        print(f"Pair: {pair[0]} and {pair[1]} - p-value: {p_value}")

results_sorted = sorted(results, key=lambda x: x[2])

# Display all pairs sorted by p-value
print("\nTop Cointegrated Pairs Sorted by p-value:")
for pair in results_sorted:
    print(f"Pair: {pair[0]} and {pair[1]} - p-value: {pair[2]}")
