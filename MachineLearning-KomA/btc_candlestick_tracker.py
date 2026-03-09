"""
Bitcoin 1-Minute Candlestick Tracker — Built from Live Trade Stream
Tracks: Open, High, Low, Close (last price before candle closes), Current
Saves candle history to candles.json for the live dashboard to read.

Note: if the code doesnt run because of Connection Closed, try using VPN or UGM WiFi.
"""

import sys
import json
import threading
import time
import os
from datetime import datetime
import websocket

# Force UTF-8 so arrow characters render correctly on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ── Config ────────────────────────────────────────────────────────────────────
CANDLE_SECONDS  = 60
OUTPUT_JSON     = os.path.join(os.path.dirname(__file__), "candles.json")
MAX_STORED      = 50   # How many closed candles to keep in the JSON file

# ── Candle State ──────────────────────────────────────────────────────────────
candle = {
    "open":       None,
    "high":       None,
    "low":        None,
    "close":      None,   # Last price recorded just before the candle closed
    "current":    None,
    "start_time": None,
}
candle_lock    = threading.Lock()
closed_candles = []   # History of completed candles


# ── Candle Helpers ────────────────────────────────────────────────────────────
def reset_candle(first_price):
    """Start a brand new candle."""
    candle["open"]       = first_price
    candle["high"]       = first_price
    candle["low"]        = first_price
    candle["close"]      = None          # Not set until candle closes
    candle["current"]    = first_price
    candle["start_time"] = time.time()


def update_candle(price):
    """Update the live candle with a new trade price."""
    candle["current"] = price
    candle["high"]    = max(candle["high"], price)
    candle["low"]     = min(candle["low"],  price)


def close_candle():
    """
    Finalize the candle:
    - 'close' = the very last price before the candle window ended
    - Save it to history and write to JSON for the dashboard
    """
    candle["close"] = candle["current"]  # Last price = close price

    closed = {
        "time":    datetime.fromtimestamp(candle["start_time"]).strftime("%H:%M"),
        "open":    candle["open"],
        "high":    candle["high"],
        "low":     candle["low"],
        "close":   candle["close"],
    }
    closed_candles.append(closed)

    # Keep only the last MAX_STORED candles
    if len(closed_candles) > MAX_STORED:
        closed_candles.pop(0)

    # Write to JSON so the dashboard can read it
    save_to_json()
    return closed


def save_to_json():
    """Write closed candle history + live candle to a JSON file."""
    now = time.time()
    elapsed   = int(now - candle["start_time"]) if candle["start_time"] else 0
    remaining = max(0, CANDLE_SECONDS - elapsed)

    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "live": {
            "time":      datetime.fromtimestamp(candle["start_time"]).strftime("%H:%M") if candle["start_time"] else "--",
            "open":      candle["open"],
            "high":      candle["high"],
            "low":       candle["low"],
            "close":     candle["close"],
            "current":   candle["current"],
            "remaining": remaining,
        },
        "history": closed_candles,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f)


def print_candle(label="live..."):
    """Print the current candle state to the terminal."""
    o = candle["open"]
    h = candle["high"]
    l = candle["low"]
    c = candle["current"]
    cl = candle["close"]

    change     = c - o
    change_pct = (change / o) * 100
    direction  = " UP" if change >= 0 else "DN"

    elapsed   = int(time.time() - candle["start_time"])
    remaining = CANDLE_SECONDS - elapsed
    timer = f"{remaining:>2}s left" if label == "live..." else label

    close_str = f"  CLOSE: ${cl:>10,.2f}" if cl is not None else ""
    ts = datetime.now().strftime("%H:%M:%S")

    print(
        f"[{ts}] "
        f"O: ${o:>10,.2f}  "
        f"H: ${h:>10,.2f}  "
        f"L: ${l:>10,.2f}  "
        f"C: ${c:>10,.2f}"
        f"{close_str}  "
        f"{'UP' if change_pct >= 0 else 'DN'} {abs(change_pct):.3f}%  "
        f"[{timer}]"
    )


# ── WebSocket Handlers ────────────────────────────────────────────────────────
def on_message(ws, message):
    data  = json.loads(message)
    price = float(data["p"])

    with candle_lock:
        now = time.time()

        if candle["start_time"] is None:
            reset_candle(price)

        elif now - candle["start_time"] >= CANDLE_SECONDS:
            # Candle expired — record close price, finalize, start new candle
            update_candle(price)           # One last update before closing
            closed = close_candle()
            print_candle(label="CLOSED ✓")
            print(
                f"  └─ CLOSE PRICE: ${closed['close']:,.2f}  |  "
                f"Range: ${closed['high'] - closed['low']:,.2f}"
            )
            print("-" * 95)
            reset_candle(price)

        else:
            update_candle(price)
            print_candle()
            save_to_json()   # Keep the JSON fresh for the dashboard


def on_open(ws):
    print("=" * 95)
    print("  BTC/USDT 1-Min Candle Tracker (Live)  |  O=Open  H=High  L=Low  C=Current  CLOSE=Final")
    print("=" * 95)
    print(f"  Stream started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dashboard data: {OUTPUT_JSON}\n")


def on_close(ws, close_status_code, close_msg):
    print("\nConnection closed. Reconnecting in 5s...")
    time.sleep(5)
    start_stream()


def on_error(ws, error):
    print(f"WebSocket error: {error}")


# ── Stream ────────────────────────────────────────────────────────────────────
def start_stream():
    stream_url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    ws = websocket.WebSocketApp(
        stream_url,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error,
    )
    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()
    return wst


if __name__ == "__main__":
    try:
        thread = start_stream()
        while thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped by user. Goodbye!")
