from pandas_datareader import data as pdr
import datetime
from datetime import datetime, timedelta
from pathlib import Path

years = 1

dt_end = datetime.today() - timedelta(days=1)
dt_start = dt_end - timedelta(days=365 * years)
data_dir = Path("capm_data")
data_dir.mkdir(exist_ok=True)

# data = pdr.get_data_yahoo('^GSPC', start='2020-01-01', end='2023-01-01')
data = pdr.get_data_yahoo('^GSPC', start=dt_start, end=dt_end)
data.to_csv(data_dir / "sp_500.csv")
