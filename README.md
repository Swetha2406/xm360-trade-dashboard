# XM360 Trade Dashboard

A live trading signal platform for XM 360 — analyses historical trends, runs backtests, and gives you exact **entry, stop loss, and take profit** values to trade.

![Dashboard Preview](https://img.shields.io/badge/status-live-brightgreen) ![Python](https://img.shields.io/badge/python-3.9+-blue) ![Flask](https://img.shields.io/badge/flask-3.0-lightgrey)

---

## What it does

| Feature | Description |
|---|---|
| **Live signals** | BUY / SELL / WAIT with exact entry, SL, TP values |
| **Price charts** | Candlestick chart with MA20, MA50, Bollinger Bands, RSI, MACD |
| **Backtest** | Test strategy on real historical data — win rate, profit factor, drawdown |
| **Sweep All** | Rank all 6 instruments at once to find the best setup |
| **My Trades** | Log your open trades with entry/SL/TP/lots |
| **Position sizing** | Calculates exact lot size based on your account and risk % |

**Instruments:** GOLD.i#, EURUSD#, GBPUSD#, USDJPY#, GBPJPY#, EURJPY#

---

## Quick start

### Windows
```
1. Install Python from https://python.org  (tick "Add to PATH")
2. Double-click start.bat
3. Open http://localhost:5000
```

### Mac / Linux
```bash
git clone https://github.com/YOUR_USERNAME/xm360-trade-dashboard.git
cd xm360-trade-dashboard
chmod +x start.sh
./start.sh
```

### Manual
```bash
git clone https://github.com/YOUR_USERNAME/xm360-trade-dashboard.git
cd xm360-trade-dashboard
pip install -r requirements.txt
python backend/server.py
# Open http://localhost:5000
```

---

## Project structure

```
xm360-trade-dashboard/
├── backend/
│   └── server.py          # Flask API — signals, backtest, positions
├── frontend/
│   └── index.html         # Full dashboard UI
├── data/
│   └── positions.json     # Your logged trades (auto-created)
├── requirements.txt
├── start.bat              # Windows one-click start
├── start.sh               # Mac/Linux one-click start
└── README.md
```

---

## How to use

### Chart & Signal tab
- Pick an instrument (GOLD, EURUSD etc.) and timeframe (H1, H4, D1)
- The signal box shows **BUY / SELL / WAIT** with exact values
- Copy the **Entry, Stop Loss, Take Profit** into XM 360
- Use the position size calculator (bottom right) to know how many lots

### Backtest tab
- Adjust balance, risk %, SL/TP multipliers
- Click **Run Backtest** to test the strategy on historical data
- Check: Win Rate > 55%, Profit Factor > 1.5, Grade A or B before trading

### Sweep All tab
- Ranks all 6 instruments at once
- Best instrument shown at top with ★
- Click any row to open its chart

### My Trades tab
- After getting a signal, click **"Log This Trade"**
- Records your entry, SL, TP, lots
- Click **CLOSE** when the trade is done

---

## Data sources

| Mode | Description |
|---|---|
| **Live** | Real historical data from Yahoo Finance (requires internet) |
| **Synthetic** | Realistic simulated data (works offline, for testing) |

The dashboard shows which mode is active via a badge on the chart.

---

## Signal strategy

The signal engine scores 8 indicators:

| Indicator | Weight | Signal |
|---|---|---|
| MA20 vs MA50 crossover | 2 | Trend direction |
| Price vs MA20 & MA50   | 1 | Trend confirmation |
| RSI (14)               | 2 | Oversold / Overbought |
| MACD histogram cross   | 3 | Momentum shift |
| MA9 vs MA20 crossover  | 2 | Short-term momentum |
| Bollinger Band touch   | 1 | Mean reversion |
| Stochastic %K          | 1 | Exhaustion signal |

A **BUY** or **SELL** is fired when 6+ points align in one direction.  
SL = 1.5 × ATR from entry. TP1 = 2.5 × ATR. TP2 = 4.0 × ATR.

---

## Risk warning

> This platform is for educational and informational purposes.  
> CFDs and Forex carry high risk. Never risk more than 1–2% per trade.  
> Past backtest performance does not guarantee future results.  
> Always set a Stop Loss before entering any trade.

---

## Requirements

- Python 3.9+
- Internet connection (for live Yahoo Finance data)
- Modern browser (Chrome, Firefox, Edge)
