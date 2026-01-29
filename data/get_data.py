import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from pathlib import Path

# -----------------------
# Settings
# -----------------------
# tickers = ["AMZN", "IBM", "MSFT", "SPY", "^GSPC", "KO", "NVDA"]
tickers = ["SPY", "^GSPC", "KO", "NVDA"]
years = 15

end = datetime.today() - timedelta(days=1)
start = end - timedelta(days=365 * years)

data_dir = Path("capm_data")
data_dir.mkdir(exist_ok=True)

# -----------------------
# Download stock + market data
# -----------------------
_start = "2010-01-01"
_end   = "2025-01-01"
# prices = yf.download(tickers, start=_start, end=_end, interval="1mo", auto_adjust=True)["Close"]

# prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]

# prices.to_csv(data_dir / "prices.csv")

# -----------------------
# Download risk-free rate (3M T-Bill from FRED)
# Symbol: DTB3 (percent, annualized)
# -----------------------
rf = pdr.DataReader("DTB3", "fred", _start, _end)
rf.to_csv(data_dir / "risk_free.csv")

print("Data downloaded and cached in ./capm_data/")
