import pandas as pd
import numpy as np

def drawdowns(returns: pd.Series) -> pd.Series:
    cr = ((1 + returns).cumprod())
    rm = cr.expanding(min_periods=1).max()
    ddown = (cr / rm) - 1
    return ddown

def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1

def annualized_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> pd.Series:
    return returns.mean() / returns.std() * np.sqrt(252)

def annualized_return(returns: pd.Series) -> pd.Series:
    return returns.mean() * 252

def annualized_volatility(returns: pd.Series) -> pd.Series:
    return returns.std() * np.sqrt(252)

def skewness(returns: pd.Series) -> pd.Series:
    return returns.skew()

def kurtosis(returns: pd.Series) -> pd.Series:
    return returns.kurt()

def max_drawdown(returns: pd.Series) -> pd.Series:
    return drawdowns(returns).min()

def print_all_metrics(returns: pd.Series):
    # print('cumulative_return:', cumulative_returns(returns))
    print('annualized_sharpe_ratio:', annualized_sharpe_ratio(returns))
    print('annualized_return:', annualized_return(returns))
    print('annualized_volatility:', annualized_volatility(returns))
    print('skewness:', skewness(returns))
    print('kurtosis:', kurtosis(returns))
    print('max_drawdown:', max_drawdown(returns))