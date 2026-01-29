import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

_start = "2015-01-01"
_end   = "2025-01-01"

# -----------------------
# Load cached data
# -----------------------
data_dir = Path("capm_data")
column_names =['Date','KO','NVDA','SPY', '^GSPC']

prices = pd.read_csv(
    data_dir / "prices.csv",
    index_col=0,    
    parse_dates=True
)

start_time = pd.to_datetime(_start)
end_time = pd.to_datetime(_end)

# Create the filter condition
time_filter = (prices.index >= start_time) & (prices.index <= end_time)
prices = prices.loc[time_filter]
print(prices)

# Apply the filter



rf = pd.read_csv(
    data_dir / "risk_free.csv",
    index_col=0,
    parse_dates=True
)


# -----------------------
# Monthly returns
# -----------------------
monthly_prices = prices.resample("M").last()
returns = monthly_prices.pct_change().dropna()
print(returns)


# -----------------------
# Risk-free: annual % → monthly decimal
# -----------------------
rf_monthly = rf.resample("M").last()
rf_monthly = rf_monthly / 100 / 12

# Align index
rf_monthly = rf_monthly.loc[returns.index]
print(rf_monthly)


# -----------------------
# Excess returns
# -----------------------
excess_returns = returns.sub(rf_monthly["DTB3"], axis=0)

market = "^GSPC"


# ===========================================
# OLS 
# ===========================================
results = []

for ticker in excess_returns.columns:
    if ticker == market:
        continue

    y = excess_returns[ticker]
    x = excess_returns[market]

    beta, alpha, r_value, p_value, std_err = stats.linregress(x, y)

    results.append({
        "Ticker": ticker,
        "Alpha (monthly)": alpha,
        "Beta": beta,
        "R_squared": r_value ** 2
    })

capm_table = pd.DataFrame(results).set_index("Ticker")
print("=== Compute Alpha & Beta (for all tickers) ===>")
print(capm_table)


# -----------------------
# Mean returns
# -----------------------
mean_rf = rf_monthly["DTB3"].mean()
mean_market = returns[market].mean()

capm_table["Expected Return (CAPM)"] = (
    mean_rf + capm_table["Beta"] * (mean_market - mean_rf)
)

# Actual mean return
capm_table["Actual Return"] = returns[capm_table.index].mean()

print("=== Compute expected returns via CAPM ===>")
print(capm_table)


# -----------------------
# Plot SML
# -----------------------
beta_range = np.linspace(0, capm_table["Beta"].max() + 0.5, 100)
sml = mean_rf + beta_range * (mean_market - mean_rf)

plt.figure(figsize=(9, 6))

# SML line
plt.plot(beta_range, sml, label="Security Market Line")

# Asset points
plt.scatter(
    capm_table["Beta"],
    capm_table["Actual Return"]
)

for ticker in capm_table.index:
    plt.text(
        capm_table.loc[ticker, "Beta"],
        capm_table.loc[ticker, "Actual Return"],
        ticker
    )

plt.xlabel("Beta")
plt.ylabel("Mean Monthly Return")
plt.title("CAPM Security Market Line")
plt.legend()
plt.grid(True)
plt.show()


# ==================================================
# CAPM regression with Alpha, Beta, R², and Beta confidence intervals
# ==================================================
import pandas as pd
import numpy as np
import statsmodels.api as sm


results = []

for ticker in excess_returns.columns:
    if ticker == market:
        continue

    y = excess_returns[ticker]
    x = excess_returns[market]

    # Add constant for alpha
    X = sm.add_constant(x)

    model = sm.OLS(y, X).fit()

    alpha = model.params["const"]
    beta = model.params[market]

    r2 = model.rsquared

    # Beta stats
    beta_se = model.bse[market]
    beta_t = model.tvalues[market]
    beta_p = model.pvalues[market]

    # 95% confidence interval for beta
    beta_ci_low, beta_ci_high = model.conf_int().loc[market]

    results.append({
        "Ticker": ticker,
        "Alpha (monthly)": alpha,
        "Beta": beta,
        "Beta StdErr": beta_se,
        "Beta t-stat": beta_t,
        "Beta p-value": beta_p,
        "Beta CI 2.5%": beta_ci_low,
        "Beta CI 97.5%": beta_ci_high,
        "R_squared": r2
    })

capm_table = pd.DataFrame(results).set_index("Ticker")

print(capm_table)

print("== Expected returns (CAPM) + actual returns ==>")
mean_rf = rf_monthly["DTB3"].mean()
mean_market = returns[market].mean()

capm_table["Expected Return (CAPM)"] = (
    mean_rf + capm_table["Beta"] * (mean_market - mean_rf)
)

capm_table["Actual Return"] = returns[capm_table.index].mean()

print(capm_table)


mean_rf = rf_monthly["DTB3"].mean()
mean_market = returns[market].mean()

capm_table["Expected Return (CAPM)"] = (
    mean_rf + capm_table["Beta"] * (mean_market - mean_rf)
)

capm_table["Actual Return"] = returns[capm_table.index].mean()

print(capm_table)


print("=== SML plot ===>")
import matplotlib.pyplot as plt

beta_range = np.linspace(0, capm_table["Beta"].max() + 0.5, 100)
sml = mean_rf + beta_range * (mean_market - mean_rf)

plt.figure(figsize=(9, 6))

# SML
plt.plot(beta_range, sml, label="Security Market Line")

# Assets
plt.scatter(capm_table["Beta"], capm_table["Actual Return"])

for ticker in capm_table.index:
    plt.text(
        capm_table.loc[ticker, "Beta"],
        capm_table.loc[ticker, "Actual Return"],
        ticker
    )

plt.xlabel("Beta")
plt.ylabel("Mean Monthly Return")
plt.title("CAPM Security Market Line (with OLS Betas)")
plt.grid(True)
plt.legend()
plt.show()

