import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import math
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def detect_frequency(returns_data):
    """
    Detect the frequency of the data (daily, weekly, monthly, etc.)
    Returns periods_per_year for annualization
    """
    if isinstance(returns_data, pd.DataFrame):
        index = returns_data.index
    else:
        index = returns_data.index
    
    if len(index) < 2:
        return 252  # Default to daily
    
    # Calculate average time difference
    time_diffs = pd.Series(index).diff().dropna()
    avg_diff = time_diffs.mean()
    
    # Determine frequency based on average difference
    if avg_diff <= pd.Timedelta(days=1):
        return 252  # Daily
    elif avg_diff <= pd.Timedelta(days=7):
        return 52   # Weekly
    elif avg_diff <= pd.Timedelta(days=31):
        return 12   # Monthly
    elif avg_diff <= pd.Timedelta(days=92):
        return 4    # Quarterly
    else:
        return 1    # Annual

def annualize_rets(r, periods_per_year=None):
    """
    Annualizes a set of returns
    """
    if periods_per_year is None:
        periods_per_year = detect_frequency(r)
    
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    if n_periods == 0:
        return 0
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year=None):
    """
    Annualizes the vol of a set of returns
    """
    if periods_per_year is None:
        periods_per_year = detect_frequency(r)
    
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year=None):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    if periods_per_year is None:
        periods_per_year = detect_frequency(r)
    
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    if ann_vol == 0:
        return 0
    return ann_ex_ret/ann_vol

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    if sigma_r == 0:
        return 0
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    if sigma_r == 0:
        return 3  # Normal kurtosis
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    returns a DataFrame with columns for
    the wealth index, 
    the previous peaks, and 
    the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

def max_drawdown(r):
    """
    Calculate maximum drawdown
    """
    dd = drawdown(r)
    return dd.Drawdown.min()

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level) if len(r) > 0 else 0
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        if len(r) == 0:
            return 0
        is_beyond = r <= -var_historic(r, level=level)
        if is_beyond.sum() == 0:
            return 0
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 matrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    try:
        weights = minimize(portfolio_vol, init_guess,
                           args=(cov,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,return_is_target),
                           bounds=bounds)
        return weights.x
    except:
        return init_guess

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        if vol == 0:
            return -np.inf
        return -(r - riskfree_rate)/vol
    
    try:
        weights = minimize(neg_sharpe, init_guess,
                           args=(riskfree_rate, er, cov), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x
    except:
        return init_guess

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False, ax=None):
    """
    Plots the multi-asset efficient frontier on a given matplotlib axis.
    """
    # If no axis is provided, create a new one
    if ax is None:
        ax = plt.gca()

    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    
    # *** CORRECTION: Plot on the provided 'ax' object ***
    ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend, ax=ax)
    
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
    return ax

def correlation_analysis(returns_df):
    """
    Perform correlation analysis and return insights
    """
    corr_matrix = returns_df.corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # High correlation threshold
                high_corr_pairs.append({
                    'asset1': corr_matrix.columns[i],
                    'asset2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    return corr_matrix, high_corr_pairs

def rolling_metrics(returns_df, window=12):
    """
    Calculate rolling metrics for trend analysis
    """
    rolling_vol = returns_df.rolling(window=window).std() * np.sqrt(detect_frequency(returns_df))
    rolling_sharpe = returns_df.rolling(window=window).apply(
        lambda x: sharpe_ratio(x, 0.03) if len(x) == window else np.nan
    )
    
    return rolling_vol, rolling_sharpe

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    periods_per_year = detect_frequency(r)
    
    ann_r = r.aggregate(annualize_rets, periods_per_year=periods_per_year)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=periods_per_year)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
    dd = r.aggregate(max_drawdown)
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

def generate_portfolio_insights(returns_df, summary_stats_df):
    """
    Generate comprehensive portfolio insights
    """
    insights = []
    
    # Performance insights
    best_performer = summary_stats_df['Annualized Return'].idxmax()
    worst_performer = summary_stats_df['Annualized Return'].idxmin()
    insights.append(f"🏆 Best performing asset: {best_performer} ({summary_stats_df.loc[best_performer, 'Annualized Return']:.2%} annual return)")
    insights.append(f"📉 Worst performing asset: {worst_performer} ({summary_stats_df.loc[worst_performer, 'Annualized Return']:.2%} annual return)")
    
    # Risk insights
    riskiest = summary_stats_df['Annualized Vol'].idxmax()
    safest = summary_stats_df['Annualized Vol'].idxmin()
    insights.append(f"⚠️ Highest risk asset: {riskiest} ({summary_stats_df.loc[riskiest, 'Annualized Vol']:.2%} volatility)")
    insights.append(f"🛡️ Lowest risk asset: {safest} ({summary_stats_df.loc[safest, 'Annualized Vol']:.2%} volatility)")
    
    # Sharpe ratio insights
    best_sharpe = summary_stats_df['Sharpe Ratio'].idxmax()
    insights.append(f"📊 Best risk-adjusted returns: {best_sharpe} (Sharpe ratio: {summary_stats_df.loc[best_sharpe, 'Sharpe Ratio']:.3f})")
    
    # Drawdown insights
    max_dd_asset = summary_stats_df['Max Drawdown'].idxmin()  # Most negative
    insights.append(f"📉 Largest drawdown: {max_dd_asset} ({summary_stats_df.loc[max_dd_asset, 'Max Drawdown']:.2%})")
    
    # Correlation insights
    corr_matrix, high_corr_pairs = correlation_analysis(returns_df)
    if high_corr_pairs:
        insights.append("🔗 High correlation pairs found:")
        for pair in high_corr_pairs[:3]:  # Show top 3
            insights.append(f"   • {pair['asset1']} & {pair['asset2']}: {pair['correlation']:.3f}")
    
    return insights