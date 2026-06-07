# Lookback horizons for point-to-point price/volume changes
CHANGE_WINDOWS = [1, 5, 10, 15, 30, 45, 90]  # trading days

# Window sizes for rolling statistical summaries
ROLLING_WINDOWS = [5, 10, 15, 30, 45, 90]  # trading days

# Base OHLCV price columns
PRICE_COLS = ["open", "close", "high", "low"]

# -----------------------------------------------------------------------
# Country / currency scope
#
# NOTE — portfolio simplification:
#   In the original production system, country of origination was NOT used
#   as an explicit filter. The model operated across all securities on the
#   brokerage's platform regardless of where they were listed.
#
#   For this public demo we restrict to US and Canadian-listed equities
#   because (a) SEC enforcement data is predominantly US-focused and
#   (b) yfinance coverage is most reliable for North American exchanges.
#   This is a data availability decision, not a modelling one.
# -----------------------------------------------------------------------
SUPPORTED_COUNTRIES = ["United States", "Canada"]
SUPPORTED_CURRENCIES = ["USD", "CAD"]

# Market cap tiers used for one-hot encoding of metadata
MARKET_CAP_BINS = [0, 300e6, 2e9, 10e9, 200e9, float("inf")]
MARKET_CAP_LABELS = ["micro", "small", "mid", "large", "mega"]

# Minimum trading days required to include a ticker in the dataset
MIN_TRADING_DAYS = 45  # roughly half of a 90-day calendar window

# Windows for order flow proxy indicators (CMF, MFI)
ORDER_FLOW_WINDOWS = [5, 10, 14, 20, 30]  # trading days
