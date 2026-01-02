# 📈 Stock Analysis & Backtesting Dashboard

A **professional desktop application** for stock market analysis,
visualization, and systematic strategy backtesting.\
Built with **Python, PyQt6, PyQtGraph, and yfinance**, focusing on clean
architecture, extensibility, and quantitative finance concepts.

This project is designed both as a **learning project** and a
**portfolio-grade application**.

------------------------------------------------------------------------

## ✨ Key Features

### 📊 Stock Analysis

-   Interactive **price charts** (Line & Candlestick)
-   Mouse hover with exact price & date
-   Zoom & rectangle selection with automatic return calculation
-   Time range selection:
    -   `1M`, `3M`, `6M`, `1Y`, `3Y`, `5Y`, `10Y`, `Max`
-   Fully resizable layout using split panes (chart, tables, analytics)

### 🗓️ Market Events

-   Visual markers for:
    -   Dividends
    -   Stock splits
    -   Earnings (where available)
-   Toggle each event type on/off
-   Markers automatically adapt to the selected time range

### 📉 Bar Charts & Detailed Metrics

-   Secondary **bar chart panel** synchronized with price chart
-   Selectable metrics via dropdown:
    -   Volume
    -   Dividends
    -   Earnings
    -   Splits
-   Numeric table displayed next to the bar chart with exact values
-   Dynamic scaling based on the active time range

### 📋 Data Tables

-   Full OHLCV history table
-   Automatically filtered by selected period
-   Sorting & scrolling enabled
-   Handles MultiIndex data cleanly

------------------------------------------------------------------------

## 🧠 Backtesting Engine

### ✅ Implemented Strategy: Buy the Dip

A configurable **mean-reversion strategy**.

#### Strategy Logic

-   Buy when price has dropped by **X %** over the last **N days**
-   Limit number of buys per day
-   Fixed investment amount per trade
-   Optional transaction fees
-   No selling (accumulation strategy)

#### Backtest Outputs

-   Number of trades
-   Total transaction fees
-   Total invested capital
-   Final portfolio value
-   Total return (%)
-   Annualized return (%)
-   Buy & Hold benchmark
-   Outperformance (total & annualized)
-   Equity curve visualization
-   Trade-by-trade transaction table

------------------------------------------------------------------------

## 🧩 Strategy System Design

-   Dedicated **Strategies tab**
-   Left panel: list of all available strategies
-   Right panel: strategy-specific configuration & results
-   Easily extensible to add new strategies

Planned / supported strategies: - Buy & Hold - Dollar-Cost Averaging
(DCA) - SMA / EMA Crossover - RSI Reversion - Bollinger Band Reversion -
Momentum (12--1) - Breakout (Donchian / Turtle)

------------------------------------------------------------------------

## 🏗️ Project Structure

    app/
    ├── data/
    │   └── loader.py              # Price, dividend, split & earnings data
    ├── backtesting/
    │   └── engine.py              # Backtesting logic & metrics
    ├── gui/
    │   ├── main_window.py
    │   ├── stocks_tab.py
    │   ├── strategies_tab.py
    │   └── components/
    │       ├── charts.py          # Line & candlestick charts
    │       └── bar_chart.py       # Metric bar charts
    ├── main.py

------------------------------------------------------------------------

## 🛠️ Installation

### Requirements

-   Python **3.10+**
-   pip or conda

### Setup

``` bash
git clone https://github.com/YOUR_USERNAME/Stock_Dashboard.git
cd Stock_Dashboard
pip install -r requirements.txt
```

### Run

``` bash
python main.py
```

------------------------------------------------------------------------

## 📌 Roadmap

-   [ ] Add more trading strategies
-   [ ] Portfolio-level backtesting (multiple assets)
-   [ ] Risk metrics (max drawdown, Sharpe ratio)
-   [ ] Export backtest results (CSV / JSON)
-   [ ] Strategy parameter optimization
-   [ ] Dark/light theme toggle

Progress is tracked via **GitHub Issues & Projects**.

------------------------------------------------------------------------

## 🎓 Motivation

This project combines: - Financial market analysis - Quantitative
strategy development - Desktop UI engineering - Clean, modular software
design

It serves as a strong **portfolio project** demonstrating applied data
science and software engineering skills.

