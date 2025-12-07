import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from itertools import product
from tqdm import tqdm

# ========== USER CONFIG ==========
symbol = "SOLUSDT"
category = "spot"
backtest_days = 7
STARTING_BALANCE = 1000
bybit_fee = 0.001
leverage = 3
agg_minutes = 57

imbalance_range = range(1, 155)
ema_range = range(2, 181, 5)
tp_range = [tp for tp in [x / 100 for x in range(1, 8)] if tp >= 0.004]  # MIN TP 0.4%

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
        resp = requests.get(url, params=params).json()
        rows = resp["result"]["list"]
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
    df = df.resample(f"{agg_minutes}min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()
    return df

def bybit_fee_fn(trade_value):
    return trade_value * bybit_fee

def calc_liq_price_long(entry_price, leverage):
    return entry_price * (1 - 1/leverage)
def calc_liq_price_short(entry_price, leverage):
    return entry_price * (1 + 1/leverage)

def run_backtest(df, imbalance_lookback, ema_len, take_profit_pct, direction):
    data = df.copy()
    data["ema"] = data["Close"].ewm(span=ema_len).mean()
    data["ema_up"] = data["ema"] > data["ema"].shift(1)
    data["ema_down"] = data["ema"] < data["ema"].shift(1)
    data["above_ema"] = data["Close"] > data["ema"]
    data["below_ema"] = data["Close"] < data["ema"]
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

        # ENTRY (only one position at a time)
        if direction == "long":
            entry_cond = (
                low <= row["imb_low"] and
                close > open_ and
                row["ema_up"] and
                row["above_ema"] and
                row["tradable"] and
                position == 0
            )
        else:
            entry_cond = (
                high >= row["imb_high"] and
                close < open_ and
                row["ema_down"] and
                row["below_ema"] and
                row["tradable"] and
                position == 0
            )

        if entry_cond and not in_liquidation:
            entry_price = close
            trade_value = balance * leverage
            qty = trade_value / entry_price
            entry_fee = bybit_fee_fn(trade_value)
            balance -= entry_fee
            if direction == "long":
                tp_price = entry_price * (1 + take_profit_pct)
                liq_price = calc_liq_price_long(entry_price, leverage)
                position = 1
            else:
                tp_price = entry_price * (1 - take_profit_pct)
                liq_price = calc_liq_price_short(entry_price, leverage)
                position = -1

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
        if position == -1 and high >= liq_price:
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
                entry_fee = bybit_fee_fn(qty * entry_price)
                net_pnl = gross - entry_fee - exit_fee
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
        # TP / EXIT (SHORT)
        if position == -1 and not in_liquidation:
            if low <= tp_price:
                exit_price = tp_price
                exit_fee = bybit_fee_fn(qty * exit_price)
                gross = (entry_price - exit_price) * qty
                entry_fee = bybit_fee_fn(qty * entry_price)
                net_pnl = gross - entry_fee - exit_fee
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
        elif position == -1:
            unrealized_pnl = (entry_price - close) * qty
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

# ==== RUN BACKTEST GRID ====
print(f"Fetching data and running optimizer on 57m bars...")
df_1m = fetch_bybit_1m(symbol, category, days=backtest_days)
df_agg = resample_candles(df_1m, agg_minutes)
df = df_agg.tail(200)

results_long = []
results_short = []
for imb, ema, tp in tqdm(
    product(imbalance_range, ema_range, tp_range),
    total=len(imbalance_range)*len(ema_range)*len(tp_range),
    desc="Param search", ncols=80
):
    results_long.append({
        "imbalance": imb, "ema": ema, "tp_pct": tp, "direction": "long", **dict(zip([
            "pnl_pct", "pnl_value", "final_balance",
            "avg_win", "avg_loss", "win_rate", "rr_ratio",
            "sharpe", "drawdown", "wins", "losses"
        ], run_backtest(df, imb, ema, tp, "long")))
    })
    results_short.append({
        "imbalance": imb, "ema": ema, "tp_pct": tp, "direction": "short", **dict(zip([
            "pnl_pct", "pnl_value", "final_balance",
            "avg_win", "avg_loss", "win_rate", "rr_ratio",
            "sharpe", "drawdown", "wins", "losses"
        ], run_backtest(df, imb, ema, tp, "short")))
    })

dfres_long = pd.DataFrame(results_long)
dfres_short = pd.DataFrame(results_short)

# Remove drawdown from results DataFrames
best_long = dfres_long.sort_values("pnl_pct", ascending=False).head(1).drop(columns=["drawdown"])
best_short = dfres_short.sort_values("pnl_pct", ascending=False).head(1).drop(columns=["drawdown"])

print("\n==================== BEST LONG PARAMETERS (57m) ====================")
print(best_long.to_string(index=False))
print("\n==================== BEST SHORT PARAMETERS (57m) ===================")
print(best_short.to_string(index=False))

# ==== COMBINED SUMMARY ====
def combine_best_rows(best_long, best_short):
    l = best_long.iloc[0]
    s = best_short.iloc[0]
    total_trades = int(l["wins"] + l["losses"] + s["wins"] + s["losses"])
    total_wins = int(l["wins"] + s["wins"])
    total_losses = int(l["losses"] + s["losses"])
    total_pnl = float(l["pnl_value"] + s["pnl_value"])
    combined_final_balance = float(l["final_balance"] + s["final_balance"] - STARTING_BALANCE)
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    avg_win = (
        (l["avg_win"] * l["wins"] + s["avg_win"] * s["wins"]) / total_wins
        if total_wins > 0 else 0
    )
    avg_loss = (
        (l["avg_loss"] * l["losses"] + s["avg_loss"] * s["losses"]) / total_losses
        if total_losses > 0 else 0
    )
    return {
        "Total Trades": total_trades,
        "Wins": total_wins,
        "Losses": total_losses,
        "Win Rate %": round(win_rate, 2),
        "Total PnL": round(total_pnl, 2),
        "Combined Final Balance": round(combined_final_balance, 2),
        "Average Win": round(avg_win, 2),
        "Average Loss": round(avg_loss, 2)
    }

combined = combine_best_rows(best_long, best_short)

print("\n============== COMBINED BEST RESULTS (LONG + SHORT) ==============")
for k, v in combined.items():
    print(f"{k}: {v}")
print("==================================================================\n")

# ==== LIVE PAPER TRADER: Both sides, only one position at a time ====

long_params = best_long.iloc[0]
short_params = best_short.iloc[0]

tradelog = []
equity = STARTING_BALANCE
position = 0
entry_price = None
entry_bar_time = None
liq_price = None
tp_price = None
qty = 0
direction = None

print(f"\n--- Live paper trader: LONG({long_params['imbalance']},{long_params['ema']},{long_params['tp_pct']}) "
      f"SHORT({short_params['imbalance']},{short_params['ema']},{short_params['tp_pct']}) | 57m ---\n")

while True:   # INFINITE LOOP, stop with Ctrl+C
    try:
        df_1m = fetch_bybit_1m(symbol, category, days=3)
        df = resample_candles(df_1m, agg_minutes)
        if len(df) < (max(long_params["imbalance"], short_params["imbalance"]) + 10):
            print("Waiting for enough bars...")
            time.sleep(2)
            continue

        data = df.copy()
        data["ema_long"] = data["Close"].ewm(span=int(long_params["ema"])).mean()
        data["ema_short"] = data["Close"].ewm(span=int(short_params["ema"])).mean()
        data["ema_up_long"] = data["ema_long"] > data["ema_long"].shift(1)
        data["ema_down_short"] = data["ema_short"] < data["ema_short"].shift(1)
        data["above_ema_long"] = data["Close"] > data["ema_long"]
        data["below_ema_short"] = data["Close"] < data["ema_short"]
        data["imb_high_short"] = data["High"].rolling(int(short_params["imbalance"])).max()
        data["imb_low_long"] = data["Low"].rolling(int(long_params["imbalance"])).min()
        data["tradable"] = True

        last_idx = len(data) - 1
        row = data.iloc[last_idx]
        open_, high, low, close = row["Open"], row["High"], row["Low"], row["Close"]

        nowstr = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

        entry_cond_long = (
            low <= row["imb_low_long"] and
            close > open_ and
            row["ema_up_long"] and
            row["above_ema_long"] and
            row["tradable"] and
            position == 0
        )
        entry_cond_short = (
            high >= row["imb_high_short"] and
            close < open_ and
            row["ema_down_short"] and
            row["below_ema_short"] and
            row["tradable"] and
            position == 0
        )

        if entry_cond_long:
            entry_price = close
            trade_value = equity * leverage
            qty = trade_value / entry_price
            entry_fee = bybit_fee_fn(trade_value)
            equity -= entry_fee
            tp_price = entry_price * (1 + float(long_params["tp_pct"]))
            liq_price = calc_liq_price_long(entry_price, leverage)
            position = 1
            entry_bar_time = row.name
            direction = "long"
            print(f"{nowstr} | ENTRY (LONG) @ {entry_price:.2f} | qty={qty:.3f} | TP={tp_price:.2f} | LIQ={liq_price:.2f}")

        elif entry_cond_short:
            entry_price = close
            trade_value = equity * leverage
            qty = trade_value / entry_price
            entry_fee = bybit_fee_fn(trade_value)
            equity -= entry_fee
            tp_price = entry_price * (1 - float(short_params["tp_pct"]))
            liq_price = calc_liq_price_short(entry_price, leverage)
            position = -1
            entry_bar_time = row.name
            direction = "short"
            print(f"{nowstr} | ENTRY (SHORT) @ {entry_price:.2f} | qty={qty:.3f} | TP={tp_price:.2f} | LIQ={liq_price:.2f}")

        # ---- EXIT & LIQUIDATION LOGIC ----
        exit_cond = False
        liq_hit = False
        if position == 1:  # LONG
            if low <= liq_price:
                exit_cond = True
                liq_hit = True
            elif high >= tp_price:
                exit_cond = True
        elif position == -1:  # SHORT
            if high >= liq_price:
                exit_cond = True
                liq_hit = True
            elif low <= tp_price:
                exit_cond = True

        if exit_cond and position != 0:
            if liq_hit:
                exit_price = liq_price
                net_pnl = -equity
                equity = 0
                status = "LIQUIDATED"
            else:
                exit_price = tp_price
                if position == 1:
                    gross = (exit_price - entry_price) * qty
                else:
                    gross = (entry_price - exit_price) * qty
                exit_fee = bybit_fee_fn(qty * exit_price)
                entry_fee = bybit_fee_fn(qty * entry_price)
                net_pnl = gross - entry_fee - exit_fee
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
            print("\n==== LIVE SUMMARY (LONG + SHORT) ====")
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
