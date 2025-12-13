# BYBIT Long Scalper (Paper Trading + Backtest)

## What this repo does
This script performs **long-only scalp research on Bybit SOLUSDT spot data**. It:
- Downloads recent candles from Bybit's public API, aggregates them to a configurable timeframe (default 5 minutes), and optimizes a simple entry/exit rule set.

- Simulates fills with spread, slippage, and order-rejection probabilities so the backtester behaves closer to the live paper trader.
- Enters one long position at a time when price dips into recent lows while staying above a rising EMA; exits at the fixed TP or marks a total loss if liquidation is reached.
- Continuously paper-trades the best-performing long parameters in an infinite loop (no real orders are sent).

## Quick start
1. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
2. **Run the optimizer + paper trader**
   ```sh
   python main.py
   ```
   - The script fetches ~`backtest_days` of 1m data, resamples to `agg_minutes`, runs the grid search, prints the best long parameters, and then starts the live paper-trading loop using those parameters.

## Core configuration (top of `main.py`)
- `symbol`/`category`: instrument to fetch (default `SOLUSDT` spot).
- `backtest_days`: historical window used to build the optimizer dataset.
- `agg_minutes`: bar size for both backtest and live loop (defaults to 5m).
- `STARTING_BALANCE`, `leverage`, `bybit_fee`: account model for PnL math.
- `spread_bps`, `slippage_bps`, `order_reject_prob`, `max_fill_latency`: live-style fill modeling shared by backtest and paper trader.

## Strategy logic
- **Entry (long-only):**
  - Price trades at/below the rolling low over `imbalance_lookback` bars.
  - Candle closes above its open and above a rising EMA (`ema_len`).
  - Only one open position at a time; new entries require a flat state and no active liquidation event.
- **Exit:**
  - Take-profit at `entry_price * (1 + SCALP_TP_PCT)`.
  - Liquidation shortcut if price trades below the simple margin-based liquidation estimate.
- **Fees & fills:**
  - Entry/exit both incur `bybit_fee` on notional.
  - Fill prices include modeled spread and Gaussian slippage; a small rejection probability blocks some fills.

## Outputs
- **Backtest report:** Best long parameter row with PnL%, win rate, RR ratio, Sharpe (drawdown removed), and trade counts for the sampled window.
- **Live paper trader logs:** Timestamped entry/exit prints, equity (realized/unrealized), and rolling totals of wins, losses, and PnL after each trade.

## Safety and compliance
- This is an **educational research tool only**. It **does not place real orders** and should not be used to manage live funds.
- If you adapt or deploy any part of this code, you are responsible for meeting all local regulatory requirements. Australian users should review [ASIC AFSL guidance](https://asic.gov.au/for-finance-professionals/afsl/).
- The model simplifies market structure (e.g., ignores funding, partial fills, tax, liquidity depth). Past performance metrics from the backtest are **not predictive** of future results.

## Known limitations
- Long-only; no short-side logic is implemented.
- Liquidation math is approximate and meant for research, not brokerage-accurate risk controls.
- The live paper trader runs indefinitely; stop with `Ctrl+C`.

---
**End of README**
