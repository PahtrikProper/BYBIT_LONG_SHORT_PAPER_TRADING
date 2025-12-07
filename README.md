# README: Quantitative Backtest and Paper Trading Script

**Compliance Notice for Australian Users**

---

## 1. Purpose and Compliance

This software is an **algorithmic trading research tool** for **educational and research purposes only**.
**It is NOT intended for live trading with real money or the management of third-party assets.**

* This script does **not** place real orders on exchanges.
* It uses only public market data (from Bybit’s public API).
* It does not constitute financial advice or a recommendation to buy/sell any asset.
* You must NOT use this software for actual trading or to provide trading signals to others without an Australian Financial Services Licence (AFSL) or explicit legal approval.

**Australian financial regulations are strict**:

* If you use, adapt, or share this script, you are responsible for ensuring compliance with all relevant laws (see: [ASIC guidance](https://asic.gov.au/for-finance-professionals/afsl/)).
* The authors accept no responsibility for any financial loss, regulatory action, or misuse.

---

## 2. Description of the Strategy

This script implements a **trend-following and mean-reversion strategy** based on historical price data:

* **Backtest Optimisation**:
  The script tests many combinations of three core parameters:

  * **Imbalance Lookback**: How many previous candles to check for price highs/lows (detecting supply/demand “imbalances”).
  * **Exponential Moving Average (EMA) Length**: Determines the trend direction.
  * **Take-Profit Percentage**: Fixed take-profit, measured as a percentage from entry price.

* **Entry Conditions**:

  * **Long trades** are entered when price dips below a recent low, closes above the open, and is above a rising EMA.
  * **Short trades** are entered when price spikes above a recent high, closes below the open, and is below a falling EMA.

* **Exits**:

  * **Take-Profit**: Trades are closed at a fixed % profit.
  * **Liquidation**: If the price moves so far against the position that it would trigger a broker liquidation (based on leverage), the trade is considered a total loss.

* **Backtest and Paper Trading**:

  * The script runs a grid search to find the best parameters over historical data, then applies those parameters in a rolling “live paper trade” mode (simulated, not live trading).

---

## 3. Functions and Required Libraries

### Python Libraries Required:

* `pandas` – Data manipulation and time-series processing.
* `numpy` – Numerical calculations.
* `requests` – HTTP API calls (fetching price data).
* `time` – Handling sleep intervals and timestamps.
* `datetime` – UTC and timestamp handling.
* `itertools.product` – Efficient parameter grid search.
* `tqdm` – Progress bars for parameter optimisation (optional for UI).

**To install:**

```sh
pip install pandas numpy requests tqdm
```

### Key Functions:

* **fetch_bybit_1m**:
  Downloads 1-minute historical OHLCV data from Bybit’s public market API for the chosen symbol and timeframe.

* **resample_candles**:
  Aggregates raw minute data into higher timeframes (default 57 minutes).

* **bybit_fee_fn**:
  Calculates trading fees based on the notional trade value and Bybit’s published rates.

* **calc_liq_price_long / calc_liq_price_short**:
  Computes the price at which a leveraged position would be liquidated (based on simple margin math, *not exact broker model*).

* **run_backtest**:
  Runs the trading strategy logic over the input data for a given parameter set, returning performance metrics (PnL, win rate, etc.).

* **combine_best_rows**:
  Sums and averages metrics for the best long and short parameter sets to give a combined strategy performance summary.

* **Live Paper Trader Block**:
  Continuously fetches new data, applies the best strategy, and simulates entries/exits in real time (simulation only, no live orders).

---

## 4. Important Disclaimers

* **This tool is not financial advice.**
* Do not use for live trading or real-money trading unless you hold an AFSL or are legally authorised.
* The models, metrics, and functions here are **simplified and do not account for all real-world factors** (e.g. slippage, funding, liquidity, partial fills, tax).
* Past performance is **not indicative of future results**.
* By using or modifying this script, **you accept full responsibility for any outcomes**.

---

**End of README**
