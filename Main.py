import pandas as pd
import numpy as np
import random
import requests
import time
from datetime import datetime
from itertools import product
from tqdm import tqdm

# ========== USER CONFIG ==========
symbol = "SOLUSDT"
category = "spot"
backtest_days = 2
STARTING_BALANCE = 1000
bybit_fee = 0.001
leverage = 10
agg_minutes = 5
spread_bps = 2  # simulated spread in basis points (0.02%)
slippage_bps = 3  # additional slippage beyond spread (avg in bps)
order_reject_prob = 0.01  # probability an order is rejected (simulated failure)
max_fill_latency = 0.5  # seconds

imbalance_range = range(1, 155)
ema_range = range(2, 181, 5)
SCALP_TP_PCT = 0.0034  # Fixed 0.34% scalp target (price move, pre-leverage)
# Do not search TP—keep the live paper trader locked to a 0.34% price move
# before leverage (≈3.4% on 10x) to avoid optimizing the take-profit.

def fetch_bybit_1m(symbol, category, limit=1000, days=backtest_days):
    end = int(datetime.utcnow().timestamp())
    start = end - days * 24 * 60 * 60
    df_list = []
    while start < end:
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": category,
            "symbol": symbol,
            "interval": "1",
            "start": start * 1000,
            "limit": limit
        }
        resp = requests.get(url, params=params, timeout=10)
        if not resp.ok:
            raise RuntimeError(f"Bybit API request failed with status {resp.status_code}: {resp.text}")
        payload = resp.json()
        if str(payload.get("retCode")) != "0":
            raise RuntimeError(f"Bybit API returned error code {payload.get('retCode')}: {payload.get('retMsg')}")

        rows = payload["result"].get("list", [])
        if not rows:
            break
        df = pd.DataFrame(rows, columns=[
            "timestamp", "Open", "High", "Low", "Close", "Volume", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.sort_values("timestamp")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
        df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
        df.set_index("timestamp", inplace=True)
        df_list.append(df)
        start = int(df.index[-1].timestamp()) + 60
        time.sleep(0.2)
    return pd.concat(df_list).sort_index()

def resample_candles(df, agg_minutes=agg_minutes):
    if df.empty:
        raise ValueError("No candle data received from Bybit; cannot resample.")

    df = df.resample(f"{agg_minutes}min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()
    if df.empty:
        raise ValueError("Aggregated dataframe is empty after resampling."
                         " Try lowering agg_minutes or increasing backtest_days.")
    return df

def bybit_fee_fn(notional):
    """Calculate Bybit taker/maker fee on executed notional at fill time."""
    return notional * bybit_fee


def simulate_order_fill(direction, mid_price, spread_bps, slippage_bps, reject_prob=0):
    """Simulate Bybit-like fills including spread, slippage, and failures.

    Returns (fill_price, status) where status is "filled" or "rejected".
    """
    if random.random() < reject_prob:
        return None, "rejected"

    spread = mid_price * (spread_bps / 10000)
    slippage = abs(np.random.normal(slippage_bps, slippage_bps / 2))
    slippage_amt = mid_price * (slippage / 10000)

    if direction == "long":
        fill_price = mid_price + spread + slippage_amt
    else:
        fill_price = mid_price - spread - slippage_amt

    time.sleep(random.uniform(0, max_fill_latency))
    return fill_price, "filled"


def log_config():
    print("\n===== PAPER TRADER CONFIGURATION =====")
    print(f"Symbol: {symbol} | Category: {category}")
    print(f"Backtest window (days): {backtest_days} | Aggregation: {agg_minutes}m")
    print(
        f"Leverage: {leverage}x | Fees: {bybit_fee * 100:.2f}% per fill on executed notional"
        " (charged separately on entry and exit fills)"
    )
    print(f"Spread model: {spread_bps} bps | Slippage model: ~{slippage_bps} bps")
    print(
        "Scalp TP focus: "
        f"{SCALP_TP_PCT * 100:.2f}% price move (pre-leverage) on ~{agg_minutes}m bars"
    )
    print(f"Order reject probability: {order_reject_prob * 100:.2f}%")
    print(f"Max simulated latency: {max_fill_latency}s")
    print("======================================\n")

def calc_liq_price_long(entry_price, leverage):
    return entry_price * (1 - 1/leverage)


def run_backtest(df, imbalance_lookback, ema_len, take_profit_pct):
    data = df.copy()
    data["ema"] = data["Close"].ewm(span=ema_len).mean()
    data["ema_up"] = data["ema"] > data["ema"].shift(1)
    data["above_ema"] = data["Close"] > data["ema"]
    data["imb_high"] = data["High"].rolling(imbalance_lookback).max()
    data["imb_low"] = data["Low"].rolling(imbalance_lookback).min()
    data["tradable"] = True

    balance = STARTING_BALANCE
    equity_curve = []
    position = 0
    entry_price = None
    tp_price = None
    liq_price = None
    qty = 0
    wins = 0
    losses = 0
    win_sizes = []
    loss_sizes = []
    in_liquidation = False

    for i in range(imbalance_lookback + 2, len(data)):
        if balance <= 0:
            equity_curve.append(0)
            continue

        row = data.iloc[i]
        open_, high, low, close = row["Open"], row["High"], row["Low"], row["Close"]

        # ENTRY (only one position at a time, long-only)
        entry_cond = (
            low <= row["imb_low"] and
            close > open_ and
            row["ema_up"] and
            row["above_ema"] and
            row["tradable"] and
            position == 0
        )

        if entry_cond and not in_liquidation:
            entry_price = close
            trade_value = balance * leverage
            qty = trade_value / entry_price
            entry_fee = bybit_fee_fn(qty * entry_price)
            balance -= entry_fee
            tp_price = entry_price * (1 + take_profit_pct)
            liq_price = calc_liq_price_long(entry_price, leverage)
            position = 1

        # LIQUIDATION
        if position == 1 and low <= liq_price:
            balance = 0
            losses += 1
            loss_sizes.append(-100)
            position = 0
            entry_price = None
            qty = 0
            in_liquidation = True
            equity_curve.append(0)
            continue

        # TP / EXIT (LONG)
        if position == 1 and not in_liquidation:
            if high >= tp_price:
                exit_price = tp_price
                exit_fee = bybit_fee_fn(qty * exit_price)
                gross = (exit_price - entry_price) * qty
                net_pnl = gross - exit_fee
                balance += net_pnl
                if net_pnl > 0:
                    wins += 1
                    win_sizes.append((net_pnl / STARTING_BALANCE) * 100)
                else:
                    losses += 1
                    loss_sizes.append((net_pnl / STARTING_BALANCE) * 100)
                position = 0
                entry_price = None
                qty = 0

        # Track equity for metrics
        if position == 1:
            unrealized_pnl = (close - entry_price) * qty
            equity = balance + unrealized_pnl
        else:
            equity = balance
        equity_curve.append(max(equity, 0))

    if not equity_curve:
        return (0, 0, STARTING_BALANCE, 0, 0, 0, None, 0, 0, 0, 0)

    final_balance = equity_curve[-1]
    pnl_value = final_balance - STARTING_BALANCE
    pnl_pct = (pnl_value / STARTING_BALANCE) * 100
    avg_win = np.mean(win_sizes) if win_sizes else 0
    avg_loss = np.mean(loss_sizes) if loss_sizes else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    rr_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else None
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365*24*60/agg_minutes) if returns.std() != 0 else 0

    # drawdown completely removed
    return (
        pnl_pct, pnl_value, final_balance,
        avg_win, avg_loss, win_rate, rr_ratio,
        sharpe, 0, wins, losses
    )

log_config()

# ==== RUN BACKTEST GRID ====
print(f"Fetching data and running optimizer on {agg_minutes}m bars...")
df_1m = fetch_bybit_1m(symbol, category, days=backtest_days)
df_agg = resample_candles(df_1m, agg_minutes)
df = df_agg.tail(200)

results_long = []
for imb, ema in tqdm(
    product(imbalance_range, ema_range),
    total=len(imbalance_range)*len(ema_range),
    desc="Param search", ncols=80
):
    results_long.append({
        "imbalance": imb, "ema": ema, "tp_pct": SCALP_TP_PCT, "direction": "long", **dict(zip([
            "pnl_pct", "pnl_value", "final_balance",
            "avg_win", "avg_loss", "win_rate", "rr_ratio",
            "sharpe", "drawdown", "wins", "losses"
        ], run_backtest(df, imb, ema, SCALP_TP_PCT)))
    })

dfres_long = pd.DataFrame(results_long)

# Remove drawdown from results DataFrames
best_long = dfres_long.sort_values("pnl_pct", ascending=False).head(1).drop(columns=["drawdown"])

print(f"\n==================== BEST LONG PARAMETERS ({agg_minutes}m) ====================")
print(best_long.to_string(index=False))

# ==== LIVE PAPER TRADER: Long-only, one position at a time ====

long_params = best_long.iloc[0]

tradelog = []
equity = STARTING_BALANCE
position = 0
entry_price = None
entry_bar_time = None
liq_price = None
tp_price = None
qty = 0
direction = None

def mark_to_market_equity(equity, position, entry_price, qty, last_price):
    if position == 1:
        return equity + (last_price - entry_price) * qty
    return equity

print(
    f"\n--- Live paper trader (scalp ~{SCALP_TP_PCT * 100:.2f}% price move pre-leverage):"
    f" LONG({long_params['imbalance']},{long_params['ema']},{long_params['tp_pct']}) | {agg_minutes}m ---\n"
)

while True:   # INFINITE LOOP, stop with Ctrl+C
    try:
        df_1m = fetch_bybit_1m(symbol, category, days=3)
        df = resample_candles(df_1m, agg_minutes)
        if len(df) < (long_params["imbalance"] + 10):
            print("Waiting for enough bars...")
            time.sleep(2)
            continue

        data = df.copy()
        data["ema_long"] = data["Close"].ewm(span=int(long_params["ema"])).mean()
        data["ema_up_long"] = data["ema_long"] > data["ema_long"].shift(1)
        data["above_ema_long"] = data["Close"] > data["ema_long"]
        data["imb_low_long"] = data["Low"].rolling(int(long_params["imbalance"])).min()
        data["tradable"] = True

        last_idx = len(data) - 1
        row = data.iloc[last_idx]
        open_, high, low, close = row["Open"], row["High"], row["Low"], row["Close"]

        nowstr = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

        long_pnl_value = float(long_params["pnl_value"])
        if long_pnl_value <= 0:
            print(f"{nowstr} | NO EDGE detected by optimizer – standing aside.")
            time.sleep(60 * agg_minutes)
            continue

        entry_cond_long = (
            low <= row["imb_low_long"] and
            close > open_ and
            row["ema_up_long"] and
            row["above_ema_long"] and
            row["tradable"] and
            position == 0
        )
        if entry_cond_long:
            entry_price, status = simulate_order_fill(
                "long", close, spread_bps, slippage_bps, order_reject_prob
            )
            if status == "rejected":
                print(f"{nowstr} | ENTRY (LONG) rejected – simulated failure (no trade)")
            else:
                trade_value = equity * leverage
                qty = trade_value / entry_price
                entry_fee = bybit_fee_fn(qty * entry_price)
                equity -= entry_fee
                tp_price = entry_price * (1 + float(long_params["tp_pct"]))
                liq_price = calc_liq_price_long(entry_price, leverage)
                position = 1
                entry_bar_time = row.name
                direction = "long"
                print(f"{nowstr} | ENTRY (LONG) @ {entry_price:.2f} | qty={qty:.3f} | TP={tp_price:.2f} | LIQ={liq_price:.2f}")
        else:
            print(f"{nowstr} | NO TRADE – waiting for a new signal.")

        # ---- EXIT & LIQUIDATION LOGIC ----
        exit_cond = False
        liq_hit = False
        if position == 1:  # LONG
            if low <= liq_price:
                exit_cond = True
                liq_hit = True
            elif high >= tp_price:
                exit_cond = True

        if exit_cond and position != 0:
            if liq_hit:
                exit_price = liq_price
                net_pnl = -equity
                equity = 0
                status = "LIQUIDATED"
            else:
                desired_exit = tp_price
                exit_price, status = simulate_order_fill(
                    "short", desired_exit, spread_bps, slippage_bps, order_reject_prob
                )
                if status == "rejected":
                    print(f"{nowstr} | EXIT SHORT rejected – simulated failure, position still open")
                    continue
                gross = (exit_price - entry_price) * qty
                exit_fee = bybit_fee_fn(qty * exit_price)
                net_pnl = gross - exit_fee
                equity += net_pnl
                status = "TP HIT" if net_pnl > 0 else "LOSS"
            tradelog.append({
                "entry_time": entry_bar_time,
                "exit_time": row.name,
                "side": direction.upper(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": qty,
                "pnl": net_pnl,
                "status": status,
                "equity": equity
            })
            print(f"{nowstr} | EXIT @ {exit_price:.2f} | {status} | NetPnL={net_pnl:.2f} | Equity={equity:.2f}")
            position = 0
            entry_price = None
            qty = 0
            liq_price = None
            tp_price = None
            direction = None

        marked_equity = mark_to_market_equity(equity, position, entry_price, qty, close)
        if position != 0:
            print(f"{nowstr} | Equity (realized/unrealized): {equity:.2f} / {marked_equity:.2f} | Trades: {len(tradelog)}")
        else:
            print(f"{nowstr} | Equity: {equity:.2f} | Trades: {len(tradelog)}")

        # ====== LIVE CUMULATIVE STATS (updated after each trade) ======
        if len(tradelog) > 0:
            trades_df = pd.DataFrame(tradelog)
            total_pnl = trades_df["pnl"].sum()
            total_trades = len(trades_df)
            wins = trades_df[trades_df["pnl"] > 0]
            losses = trades_df[trades_df["pnl"] <= 0]
            win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
            avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
            avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
            print("\n==== LIVE SUMMARY (LONG ONLY) ====")
            print(f"Total trades: {total_trades}")
            print(f"Wins: {len(wins)} | Losses: {len(losses)}")
            print(f"Win rate: {win_rate:.2f}%")
            print(f"Total PnL: {total_pnl:.2f}")
            print(f"Average win: {avg_win:.2f}")
            print(f"Average loss: {avg_loss:.2f}")
            print("=====================================\n")

        time.sleep(60 * agg_minutes)   # Sleep until next bar (real time)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        break
    except Exception as e:
        print("Exception:", e)
        time.sleep(2)
