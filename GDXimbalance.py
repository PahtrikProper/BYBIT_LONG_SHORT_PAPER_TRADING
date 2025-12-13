import pandas as pd
import numpy as np
import random
import requests
import time
from datetime import datetime
from tqdm import tqdm

# ========== USER CONFIG ==========
symbol = "SOLUSDT"
category = "spot"
backtest_days = 2
STARTING_BALANCE = 1000
bybit_fee = 0.001
leverage = 10
timeframes = [5, 15]
spread_bps = 2  # simulated spread in basis points (0.02%)
slippage_bps = 3  # additional slippage beyond spread (avg in bps)
order_reject_prob = 0.01  # probability an order is rejected (simulated failure)
max_fill_latency = 0.5  # seconds

imbalance_lookback = 18
ema_len = 41
take_profit_pct = 0.15  # 15% target (price move)


def _parse_candles(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df = df.sort_values("timestamp")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df.set_index("timestamp", inplace=True)
    return df


def fetch_bybit_candles(symbol, category, interval_minutes, limit=1000, days=backtest_days, latest_only=False):
    """Fetch raw Bybit candles at the requested interval (no resampling)."""

    interval = str(interval_minutes)
    url = "https://api.bybit.com/v5/market/kline"

    if latest_only:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=10)
        if not resp.ok:
            raise RuntimeError(f"Bybit API request failed with status {resp.status_code}: {resp.text}")
        payload = resp.json()
        if str(payload.get("retCode")) != "0":
            raise RuntimeError(f"Bybit API returned error code {payload.get('retCode')}: {payload.get('retMsg')}")
        rows = payload["result"].get("list", [])
        if not rows:
            raise ValueError(f"No candle data received from Bybit for {interval}-minute interval.")
        return _parse_candles(rows)

    end = int(datetime.utcnow().timestamp())
    start = end - days * 24 * 60 * 60
    step = interval_minutes * 60
    df_list = []

    while start < end:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": start * 1000,
            "limit": limit,
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
        df_list.append(_parse_candles(rows))
        last_ts_ms = int(rows[-1][0])
        start = int(pd.to_datetime(last_ts_ms, unit="ms").timestamp()) + step
        time.sleep(0.2)

    if not df_list:
        raise ValueError(f"No candle data received from Bybit for {interval}-minute interval.")
    return pd.concat(df_list).sort_index()


def bybit_fee_fn(notional):
    """Calculate Bybit taker/maker fee on executed notional at fill time."""
    return notional * bybit_fee


def simulate_order_fill(direction, mid_price, spread_bps, slippage_bps, reject_prob=0, wait_latency=True):
    """Simulate Bybit-like fills including spread, slippage, and failures."""
    if random.random() < reject_prob:
        return None, "rejected"

    spread = mid_price * (spread_bps / 10000)
    slippage = abs(np.random.normal(slippage_bps, slippage_bps / 2))
    slippage_amt = mid_price * (slippage / 10000)

    if direction == "long":
        fill_price = mid_price + spread + slippage_amt
    else:
        fill_price = mid_price - spread - slippage_amt

    if wait_latency:
        time.sleep(random.uniform(0, max_fill_latency))
    return fill_price, "filled"


def log_config():
    print("\n===== GDX Imbalance Strategy CONFIGURATION =====")
    print(f"Symbol: {symbol} | Category: {category}")
    print(f"Backtest window (days): {backtest_days} | Timeframes: {timeframes}")
    print(f"Imbalance lookback: {imbalance_lookback} | EMA length: {ema_len} | TP: {take_profit_pct * 100:.2f}%")
    print(
        f"Leverage: {leverage}x | Fees: {bybit_fee * 100:.2f}% per fill on executed notional"
        " (charged separately on entry and exit fills)"
    )
    print(f"Spread model: {spread_bps} bps | Slippage model: ~{slippage_bps} bps")
    print(f"Order reject probability: {order_reject_prob * 100:.2f}%")
    print(f"Max simulated latency: {max_fill_latency}s")
    print("==============================================\n")


def run_backtest(df, interval_minutes):
    data = df.copy()
    data["ema"] = data["Close"].ewm(span=ema_len).mean()
    data["ema_up"] = data["ema"] > data["ema"].shift(1)
    data["ema_down"] = data["ema"] < data["ema"].shift(1)
    data["price_above_ema"] = data["Close"] > data["ema"]
    data["price_below_ema"] = data["Close"] < data["ema"]
    data["imb_low"] = data["Low"].rolling(imbalance_lookback).min()
    data["imb_high"] = data["High"].rolling(imbalance_lookback).max()

    balance = STARTING_BALANCE
    equity_curve = []
    position = 0
    entry_price = None
    tp_price = None
    qty = 0
    wins = 0
    losses = 0
    win_sizes = []
    loss_sizes = []

    for i in range(imbalance_lookback + 2, len(data)):
        row = data.iloc[i]
        if row.name.year != 2025:
            equity_curve.append(balance)
            continue

        open_, high, low, close = row["Open"], row["High"], row["Low"], row["Close"]

        bullish_confirm = low <= row["imb_low"] and close > open_
        long_condition = (
            bullish_confirm and row["ema_up"] and row["price_above_ema"] and position == 0
        )

        bearish_confirm = high >= row["imb_high"] and close < open_
        exit_condition = (
            bearish_confirm and row["ema_down"] and row["price_below_ema"] and position > 0
        )

        if long_condition:
            entry_price, status = simulate_order_fill(
                "long", close, spread_bps, slippage_bps, order_reject_prob, wait_latency=False
            )
            if status == "rejected":
                equity_curve.append(balance)
                continue
            trade_value = balance * leverage
            qty = trade_value / entry_price
            entry_fee = bybit_fee_fn(qty * entry_price)
            balance -= entry_fee
            tp_price = entry_price * (1 + take_profit_pct)
            position = 1

        if position == 1 and tp_price is not None and high >= tp_price:
            desired_exit = tp_price
            exit_price, status = simulate_order_fill(
                "short", desired_exit, spread_bps, slippage_bps, order_reject_prob, wait_latency=False
            )
            if status != "rejected":
                exit_fee = bybit_fee_fn(qty * exit_price)
                gross = (exit_price - entry_price) * qty
                net_pnl = gross - exit_fee
                balance += net_pnl
                wins += 1 if net_pnl > 0 else 0
                losses += 1 if net_pnl <= 0 else 0
                (win_sizes if net_pnl > 0 else loss_sizes).append((net_pnl / STARTING_BALANCE) * 100)
                position = 0
                entry_price = None
                qty = 0
                tp_price = None

        if position == 1 and exit_condition:
            exit_price, status = simulate_order_fill(
                "short", close, spread_bps, slippage_bps, order_reject_prob, wait_latency=False
            )
            if status != "rejected":
                exit_fee = bybit_fee_fn(qty * exit_price)
                gross = (exit_price - entry_price) * qty
                net_pnl = gross - exit_fee
                balance += net_pnl
                wins += 1 if net_pnl > 0 else 0
                losses += 1 if net_pnl <= 0 else 0
                (win_sizes if net_pnl > 0 else loss_sizes).append((net_pnl / STARTING_BALANCE) * 100)
                position = 0
                entry_price = None
                qty = 0
                tp_price = None

        equity = balance
        if position == 1 and entry_price is not None:
            equity += (close - entry_price) * qty
        equity_curve.append(max(equity, 0))

    if not equity_curve:
        return (0, 0, STARTING_BALANCE, 0, 0, 0, None, 0, 0, 0)

    final_balance = equity_curve[-1]
    pnl_value = final_balance - STARTING_BALANCE
    pnl_pct = (pnl_value / STARTING_BALANCE) * 100
    avg_win = np.mean(win_sizes) if win_sizes else 0
    avg_loss = np.mean(loss_sizes) if loss_sizes else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    rr_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else None
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24 * 60 / interval_minutes) if returns.std() != 0 else 0

    return (
        pnl_pct, pnl_value, final_balance,
        avg_win, avg_loss, win_rate, rr_ratio,
        sharpe, wins, losses
    )


def summarize_trades(tradelog):
    if not tradelog:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    trades_df = pd.DataFrame(tradelog)
    total_pnl = trades_df["pnl"].sum()
    total_trades = len(trades_df)
    wins_df = trades_df[trades_df["pnl"] > 0]
    losses_df = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(wins_df) / total_trades * 100 if total_trades > 0 else 0.0
    avg_win = wins_df["pnl"].mean() if len(wins_df) > 0 else 0.0
    avg_loss = losses_df["pnl"].mean() if len(losses_df) > 0 else 0.0

    return {
        "total": total_trades,
        "wins": len(wins_df),
        "losses": len(losses_df),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


log_config()

print("\n--- Backtesting GDX Imbalance Strategy ---\n")
backtest_results = {}
for tf in timeframes:
    print(f"Fetching {tf}m data...")
    df_tf = fetch_bybit_candles(symbol, category, tf, days=backtest_days)
    df_tf = df_tf.tail(200)
    metrics = run_backtest(df_tf, tf)
    backtest_results[tf] = metrics
    labels = [
        "pnl_pct", "pnl_value", "final_balance",
        "avg_win", "avg_loss", "win_rate", "rr_ratio",
        "sharpe", "wins", "losses"
    ]
    print(f"Results {tf}m:")
    for name, val in zip(labels, metrics):
        print(f"  {name}: {val}")

# ==== LIVE PAPER TRADER: Long-only per timeframe ====
state = {
    tf: {
        "tradelog": [],
        "equity": STARTING_BALANCE,
        "position": 0,
        "entry_price": None,
        "entry_bar_time": None,
        "last_bar_time": None,
        "tp_price": None,
        "qty": 0,
        "direction": None,
    }
    for tf in timeframes
}


def mark_to_market_equity(equity, position, entry_price, qty, last_price):
    if position == 1 and entry_price is not None:
        return equity + (last_price - entry_price) * qty
    return equity


print("\n--- Live paper trader running GDX Imbalance Strategy ---\n")
sleep_seconds = min(timeframes) * 60

while True:
    try:
        nowstr = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        for tf in timeframes:
            tf_state = state[tf]
            lookback_bars = max(imbalance_lookback, ema_len) + 50
            df = fetch_bybit_candles(symbol, category, tf, limit=lookback_bars, latest_only=True)
            if len(df) < (imbalance_lookback + 10):
                print(f"{nowstr} | {tf}m | Waiting for enough bars...")
                continue

            data = df.copy()
            data["ema"] = data["Close"].ewm(span=ema_len).mean()
            data["ema_up"] = data["ema"] > data["ema"].shift(1)
            data["ema_down"] = data["ema"] < data["ema"].shift(1)
            data["price_above_ema"] = data["Close"] > data["ema"]
            data["price_below_ema"] = data["Close"] < data["ema"]
            data["imb_low"] = data["Low"].rolling(imbalance_lookback).min()
            data["imb_high"] = data["High"].rolling(imbalance_lookback).max()

            row = data.iloc[-1]
            open_, high, low, close = row["Open"], row["High"], row["Low"], row["Close"]

            if tf_state["last_bar_time"] is not None and tf_state["last_bar_time"] == row.name:
                print(f"{nowstr} | {tf}m | Latest bar already processed – waiting for close.")
                continue
            tf_state["last_bar_time"] = row.name

            if row.name.year != 2025:
                print(f"{nowstr} | {tf}m | Outside 2025 window – standing aside.")
                continue

            bullish_confirm = low <= row["imb_low"] and close > open_
            long_condition = (
                bullish_confirm and row["ema_up"] and row["price_above_ema"] and tf_state["position"] == 0
            )

            bearish_confirm = high >= row["imb_high"] and close < open_
            exit_condition = (
                bearish_confirm and row["ema_down"] and row["price_below_ema"] and tf_state["position"] > 0
            )

            if long_condition:
                entry_price, status = simulate_order_fill(
                    "long", close, spread_bps, slippage_bps, order_reject_prob
                )
                if status == "rejected":
                    print(f"{nowstr} | {tf}m | ENTRY (LONG) rejected – simulated failure (no trade)")
                else:
                    trade_value = tf_state["equity"] * leverage
                    qty = trade_value / entry_price
                    entry_fee = bybit_fee_fn(qty * entry_price)
                    tf_state["equity"] -= entry_fee
                    tf_state["tp_price"] = entry_price * (1 + take_profit_pct)
                    tf_state["position"] = 1
                    tf_state["entry_bar_time"] = row.name
                    tf_state["direction"] = "long"
                    tf_state["entry_price"] = entry_price
                    tf_state["qty"] = qty
                    print(
                        f"{nowstr} | {tf}m | ENTRY (LONG) @ {entry_price:.2f} | qty={qty:.3f} "
                        f"| TP={tf_state['tp_price']:.2f}"
                    )
            else:
                print(f"{nowstr} | {tf}m | NO TRADE – waiting for a new signal.")

            exit_cond = False
            tp_hit = False
            if tf_state["position"] == 1:
                if tf_state["tp_price"] is not None and high >= tf_state["tp_price"]:
                    exit_cond = True
                    tp_hit = True
                elif exit_condition:
                    exit_cond = True

            if exit_cond and tf_state["position"] != 0:
                if tp_hit:
                    desired_exit = tf_state["tp_price"]
                else:
                    desired_exit = close
                exit_price, status = simulate_order_fill(
                    "short", desired_exit, spread_bps, slippage_bps, order_reject_prob, wait_latency=False
                )
                if status != "rejected":
                    gross = (exit_price - tf_state["entry_price"]) * tf_state["qty"]
                    exit_fee = bybit_fee_fn(tf_state["qty"] * exit_price)
                    net_pnl = gross - exit_fee
                    tf_state["equity"] += net_pnl
                    status_label = "Take Profit" if tp_hit else "Opposite Signal"
                    tf_state["tradelog"].append({
                        "entry_time": tf_state["entry_bar_time"],
                        "exit_time": row.name,
                        "side": tf_state["direction"].upper(),
                        "entry_price": tf_state["entry_price"],
                        "exit_price": exit_price,
                        "qty": tf_state["qty"],
                        "pnl": net_pnl,
                        "status": status_label,
                        "equity": tf_state["equity"],
                    })
                    print(
                        f"{nowstr} | {tf}m | EXIT @ {exit_price:.2f} | {status_label} | NetPnL={net_pnl:.2f} "
                        f"| Equity={tf_state['equity']:.2f}"
                    )
                    tf_state["position"] = 0
                    tf_state["entry_price"] = None
                    tf_state["qty"] = 0
                    tf_state["tp_price"] = None
                    tf_state["direction"] = None

            marked_equity = mark_to_market_equity(
                tf_state["equity"], tf_state["position"], tf_state["entry_price"], tf_state["qty"], close
            )
            if tf_state["position"] != 0:
                print(
                    f"{nowstr} | {tf}m | Equity (realized/unrealized): {tf_state['equity']:.2f} / "
                    f"{marked_equity:.2f} | Trades: {len(tf_state['tradelog'])}"
                )
            else:
                print(f"{nowstr} | {tf}m | Equity: {tf_state['equity']:.2f} | Trades: {len(tf_state['tradelog'])}")

            stats = summarize_trades(tf_state["tradelog"])
            print(
                f"{nowstr} | {tf}m | Price={close:.4f} | Trades={stats['total']} | "
                f"WR={stats['win_rate']:.2f}% | RealizedPnL={stats['total_pnl']:.2f} | "
                f"AvgWin={stats['avg_win']:.2f} | AvgLoss={stats['avg_loss']:.2f}"
            )

        time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        break
    except Exception as e:
        print("Exception:", e)
        time.sleep(2)
