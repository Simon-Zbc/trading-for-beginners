# trading-for-beginners(learning quantitative trading from zero)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Learning Roadmap](#learning-roadmap)
- [Phase 1 — Financial Fundamentals + Data](#phase-1--financial-fundamentals--data-week-14)
- [Phase 2 — Strategy Design + Backtesting](#phase-2--strategy-design--backtesting-week-58)
- [Phase 3 — ML/AI Integration](#phase-3--mlai-integration-week-912)
- [Phase 4 — Live Trading + Agent](#phase-4--live-trading--agent-week-1316)
- [Key Concepts Reference](#key-concepts-reference)
- [Lessons Learned](#lessons-learned)
- [Environment Setup](#environment-setup)
- [Progress Tracker](#progress-tracker)

---

## Project Structure

```
trading-for-beginners/
├── README.md
├── data/                          # Locally cached stock data
├── notebooks/
│   ├── week1_data_viz/
│   │   ├── day1_candlestick.ipynb
│   │   ├── day2_sma_ema.ipynb
│   │   ├── day3_indicators.ipynb
│   │   └── day4_analysis_function.ipynb
│   ├── week2_backtest/
│   │   ├── day1_manual_backtest.ipynb
│   │   ├── day2_costs_sharpe.ipynb
│   │   ├── day3_drawdown_longterm.ipynb
│   │   ├── day4_vectorbt_optimize.ipynb    ← in progress
│   │   └── day5_report.ipynb
│   ├── week3_indicators/
│   ├── week4_data_pipeline/
│   ├── week5_8_strategy/
│   ├── week9_12_ml/
│   └── week13_16_livetrading/
├── src/
│   ├── data.py                    # Data fetching utilities
│   ├── indicators.py              # Technical indicator library
│   ├── backtest.py                # Backtesting engine
│   └── analysis.py                # Report generation
└── results/                       # Backtest report outputs
```

---

## Learning Roadmap

```
Phase 1 (Week 1–4)       Phase 2 (Week 5–8)      Phase 3 (Week 9–12)     Phase 4 (Week 13–16)
Fundamentals + Data  →   Strategy + Backtest  →   ML/AI Integration  →   Live Trading + Agent

Milestone:               Milestone:               Milestone:               Milestone:
Read candlestick charts  First backtest strategy  AI prediction model      Ready for live trading
```

---

## Phase 1 — Financial Fundamentals + Data (Week 1–4)

### ✅ Week 1 — Candlesticks, Moving Averages, Technical Indicators (Complete)

#### Day 0 — yfinance basics

**Topics covered:**

- TSE ticker format: `number.T` (e.g. `7203.T` = Toyota)
- Downloading historical data with yfinance
- Closing price line chart

**Core code:**

```python
import yfinance as yf

df = yf.download("7203.T", start="2024-09-01", end="2025-03-01")
df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
df["Close"].plot(title="Toyota 7203.T")
```

**Key notes:**

- yfinance returns MultiIndex columns — always `droplevel(1)` to flatten
- TSE trading hours: 9:00–15:30 JST

---

#### Day 1 — Candlestick charts

**Topics covered:**

- Candlestick anatomy: open, close, high, low, wicks
- mplfinance for professional K-line charts with volume

**Core code:**

```python
import mplfinance as mpf

mpf.plot(df, type='candle', volume=True, style='yahoo',
         title='Toyota 7203.T')
```

**Known issue:** `legend_loc` parameter unsupported in some mplfinance versions — remove it.

| Color        | Meaning                 |
| ------------ | ----------------------- |
| Red candle   | Close < Open (down day) |
| Green candle | Close > Open (up day)   |

---

#### Day 2 — SMA / EMA from scratch

**Topics covered:**

- SMA: equal-weight average of past N closing prices
- EMA: exponentially weighted average, more weight on recent data
- Why `rolling()` produces NaN for the first N−1 rows

**Formulas:**

```
SMA(N) = sum of past N closes / N

EMA(today) = price(today) × α + EMA(yesterday) × (1 − α)
α = 2 / (N + 1)
```

**Validation result:**

- Hand-written SMA vs `pandas rolling().mean()` — max error: `0.0000000000` ✓

**Key insight:** EMA reacts faster than SMA during sharp price moves, catching trend reversals earlier.

---

#### Day 3 — RSI / MACD + multi-panel chart

**Topics covered:**

- `ta` library for one-line indicator calculation
- RSI (14-day): >70 overbought, <30 oversold
- MACD: fast line / signal line / histogram
- mplfinance multi-panel: candlestick + volume + RSI + MACD

**Core code:**

```python
import ta

df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
macd = ta.trend.MACD(df["Close"])
df["MACD"]        = macd.macd()
df["MACD_signal"] = macd.macd_signal()
df["MACD_hist"]   = macd.macd_diff()
```

---

#### Day 4 — Signal detection + reusable analysis function

**Topics covered:**

- Golden cross / death cross detection (SMA5 crossing SMA25)
- Encapsulating everything into `analyze_stock(ticker)`
- Cross-stock comparison across 3 TSE names

**Signal logic:**

```python
prev = df["SMA5"].shift(1) - df["SMA25"].shift(1)
curr = df["SMA5"] - df["SMA25"]
golden_cross = (prev < 0) & (curr > 0)   # Buy signal
death_cross  = (prev > 0) & (curr < 0)   # Sell signal
```

**3-stock comparison results (2024-09 ~ 2025-03):**

| Ticker | Name       | Return | Ann. Volatility | RSI  | Signals |
| ------ | ---------- | ------ | --------------- | ---- | ------- |
| 7203.T | Toyota     | −1.9%  | 33.5%           | 39.4 | 5       |
| 9984.T | SoftBank G | −2.2%  | 45.4%           | 31.7 | 9       |
| 6758.T | Sony G     | +33.2% | 30.9%           | 60.7 | 4       |

**Key insights:**

- SoftBank had the most signals (9) but worst returns → **more signals ≠ better strategy**
- Sony had a clear trend, fewer but higher-quality signals → moving average strategies only work in trending markets

---

### 📋 Week 2 — Backtesting Framework + Performance Evaluation (In Progress)

#### ✅ Day 1 — Manual backtesting engine

**Topics covered:**

- The 4 core steps of a backtest: loop each day → check signal → execute trade → record equity
- Buy & Hold as a benchmark
- Equity curve visualization

**Sony 6758.T backtest result (2024-09 ~ 2025-03):**

```
Initial capital  : ¥1,000,000
Final equity     : ¥1,087,394
Strategy return  : +21.1%
Buy & Hold       : +33.2%
Alpha            : −12.1%
```

**Key insight:** Strategy sold early due to lagging moving averages, missing a portion of the uptrend.

---

#### ✅ Day 2 — Transaction costs, slippage, and Sharpe ratio

**Topics covered:**

- Real TSE trading costs: commission 0.055% + slippage 0.05% ≈ 0.1% per trade
- Sharpe ratio = (strategy return − risk-free rate) / volatility × √252
- Japan risk-free rate: ~0.1% (10-year JGB)
- Max drawdown preview

**Result with costs (Sony, 6 months):**

```
Strategy return  : +20.69%
Total commission : ¥1,740  (only 3 trades — negligible impact)
Sharpe ratio     : 1.852   (excellent)
Max drawdown     : −6.74%
```

**Sharpe ratio benchmarks:**

| Value     | Interpretation          |
| --------- | ----------------------- |
| > 1.0     | Excellent               |
| 0.5 – 1.0 | Average fund level      |
| 0 – 0.5   | Mediocre                |
| < 0       | Worse than holding cash |

---

#### ✅ Day 3 — Max drawdown deep dive + long-horizon validation

**Topics covered:**

- Max drawdown = largest peak-to-trough decline in equity
- Drawdown formula: `(current − rolling_max) / rolling_max`
- 6-month vs 3-year comparison on the same strategy

**Key formula:**

```python
rolling_max  = result["total"].cummax()
drawdown     = (result["total"] - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100
```

**6-month vs 3-year comparison (Sony 6758.T):**

| Metric          | 6 months | 3 years |
| --------------- | -------- | ------- |
| Strategy return | +20.7%   | −16.7%  |
| Buy & Hold      | +33.2%   | +27.4%  |
| Alpha           | −12.5%   | −44.1%  |
| Sharpe ratio    | 1.9      | −0.2    |
| Max drawdown    | −6.7%    | −39.4%  |
| Win rate        | 100%     | 26.1%   |
| No. of trades   | 3        | 47      |

**⚠️ Critical lesson:**

> The 6-month results were data coincidence — the strategy happened to match an uptrend.
> Over 3 years, 47 trades with only 26% win rate → net loss of −16.7%.
> **Any strategy must be validated on at least 3 years of historical data.**

---

#### 🔲 Day 4 — vectorbt parameter optimization (upcoming)

**Planned topics:**

- Install and learn the vectorbt framework
- Sweep 100+ moving average combinations on 3-year data
- Plot parameter heatmap to find robust settings

---

#### 🔲 Day 5 — Multi-stock backtest report (upcoming)

**Planned topics:**

- Run backtest simultaneously on TSE Top 10 stocks
- Auto-generate HTML performance report
- Cross-strategy ranking

---

### 🔲 Week 3 — Additional Technical Indicators

| Day   | Topic                                       |
| ----- | ------------------------------------------- |
| Day 1 | Bollinger Bands — theory and implementation |
| Day 2 | Bollinger Band breakout strategy backtest   |
| Day 3 | Volume indicators: OBV, VWAP                |
| Day 4 | Multi-indicator confirmation signals        |
| Day 5 | TSE sector rotation analysis                |

---

### 🔲 Week 4 — Data Pipeline

| Day   | Topic                                             |
| ----- | ------------------------------------------------- |
| Day 1 | Missing value handling, split/dividend adjustment |
| Day 2 | Batch download TSE Top 100                        |
| Day 3 | Local database with SQLite                        |
| Day 4 | Automated update script (cron / scheduler)        |
| Day 5 | Data quality validation pipeline                  |

---

## Phase 2 — Strategy Design + Backtesting (Week 5–8)

### 🔲 Week 5 — Advanced Backtesting

| Day   | Topic                                              |
| ----- | -------------------------------------------------- |
| Day 1 | Walk-forward analysis (preventing overfitting)     |
| Day 2 | Position sizing: fixed fraction vs Kelly criterion |
| Day 3 | Stop-loss and take-profit logic                    |
| Day 4 | Multi-stock portfolio backtesting                  |
| Day 5 | Automated strategy report generation               |

### 🔲 Week 6–8 — Strategy Development

- RSI overbought/oversold strategy
- MACD crossover strategy
- Bollinger Band breakout strategy
- Multi-indicator combined strategy
- Cross-strategy performance comparison

---

## Phase 3 — ML/AI Integration (Week 9–12)

> **This phase directly leverages your existing AI development background**

### 🔲 Week 9 — Feature Engineering

- Unique properties of financial time series (non-stationarity, noise)
- Technical indicators → ML features (preventing look-ahead bias)
- Label design: defining the prediction target (up/down/neutral)
- Time series cross-validation with `TimeSeriesSplit`

### 🔲 Week 10 — Machine Learning Models

- XGBoost classifier for next-day direction prediction
- Feature importance analysis
- Evaluation metrics (precision, recall, F1)

### 🔲 Week 11 — Deep Learning + Sentiment Analysis

- LSTM for price sequence forecasting
- Japanese news sentiment analysis (leveraging RAG/NLP experience)
- Alternative data source exploration

### 🔲 Week 12 — Signal Fusion

- Technical signal + ML signal combination
- Full AI strategy backtest
- Comparison against traditional strategies

---

## Phase 4 — Live Trading + Agent (Week 13–16)

### 🔲 Week 13 — Broker API Integration

- SBI Securities / Rakuten Securities API research
- Paper trading (simulated account) setup
- Real-time data stream handling

### 🔲 Week 14 — Execution Automation

- Scheduled jobs (cron / APScheduler)
- Alert notifications (LINE / Slack)
- Error handling and logging system

### 🔲 Week 15 — Trading Agent

> **Leverages existing Agent development experience**

- Trading decision Agent architecture
- Tool calls: data fetch → signal generation → order placement → monitoring
- Multi-agent collaboration (Analysis Agent + Execution Agent + Risk Agent)

### 🔲 Week 16 — Risk Management + Go Live

- Position risk controls (max single-stock exposure, total capital limit)
- Anomaly detection (price spikes, API failures)
- Small capital live deployment
- Continuous monitoring and iteration

---

## Key Concepts Reference

### Technical Indicators

| Indicator | Parameters           | Purpose                 | Signal                              |
| --------- | -------------------- | ----------------------- | ----------------------------------- |
| SMA       | Window N             | Trend direction         | Golden cross buy / death cross sell |
| EMA       | Span N (α = 2/(N+1)) | Trend (more responsive) | Same as SMA                         |
| RSI       | 14-day, range 0–100  | Overbought / oversold   | >70 overbought / <30 oversold       |
| MACD      | 12 / 26 / 9          | Trend momentum          | Signal line crossover               |
| Bollinger | 20-day, ±2σ          | Volatility breakout     | Price outside band                  |

### Performance Metrics

| Metric          | Formula                              | Benchmark                     |
| --------------- | ------------------------------------ | ----------------------------- |
| Return          | (final / initial − 1) × 100          | Beat Buy & Hold               |
| Sharpe ratio    | Ann. excess return / Ann. volatility | > 1.0 excellent               |
| Max drawdown    | (trough − peak) / peak               | < 20% acceptable              |
| Win rate        | Winning trades / total trades        | MA strategies: 30–50% typical |
| Ann. volatility | Daily return std × √252              | < 30% relatively stable       |

### TSE Trading Rules

| Item            | Detail                                |
| --------------- | ------------------------------------- |
| Trading hours   | 9:00–15:30 JST                        |
| Ticker format   | `number.T` (yfinance)                 |
| Minimum lot     | 100 shares (単元株)                   |
| Commission      | ~0.055% (SBI / Rakuten, tax included) |
| Fiscal year-end | March / September most common         |

---

## Lessons Learned

### 📌 Lesson 1: More signals ≠ better strategy

SoftBank (9984) generated 9 signals in 6 months.
High signal frequency means the strategy is repeatedly triggered in choppy markets — commissions pile up and false signal ratio is high.
→ **A good strategy should have fewer, higher-quality signals.**

### 📌 Lesson 2: Short-term backtests are overfitting traps

Sony 6-month backtest: Sharpe 1.9, return +20.7% — looks great.
Same strategy over 3 years: Sharpe −0.2, return −16.7% — completely broken.
→ **Every strategy must be validated on at least 3 years of historical data.**

### 📌 Lesson 3: Moving average strategies only work in trending markets

Over 3 years: 47 trades, win rate only 26%.
In sideways / choppy markets, price repeatedly crosses the moving average, generating floods of false signals.
→ **Need to add a trend filter, or switch to a mean-reversion strategy for range-bound conditions.**

### 📌 Lesson 4: Max drawdown is the most important risk metric

A −39.4% max drawdown means the account was down nearly 40% at its worst.
Most people panic-close at the bottom — real-world returns are far worse than backtest results.
→ **Controlling drawdown is more important than maximizing return.**

### 📌 Lesson 5: Commission impact scales with trade frequency

3 trades: ¥1,740 commission — negligible (0.17% impact)
47 trades: ~¥27,000+ — meaningful drag
→ **Low-turnover strategies have a natural cost advantage.**

---

## Environment Setup

### Install dependencies

```bash
pip install yfinance pandas matplotlib mplfinance ta vectorbt jupyter notebook
```

### Version notes

```
Python     : 3.10+
yfinance   : 0.2.x
mplfinance : 0.12.x  (legend_loc unsupported in some versions — remove it)
ta         : 0.10.x
vectorbt   : 0.26.x
```

### Data fetch template

```python
import yfinance as yf

def get_stock(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
    df.dropna(inplace=True)
    return df

# Key TSE tickers
TICKERS = {
    "7203.T": "Toyota",
    "9984.T": "SoftBank Group",
    "6758.T": "Sony Group",
    "6861.T": "Keyence",
    "8306.T": "Mitsubishi UFJ",
}
```

---

## Progress Tracker

- [x] Week 1 — Financial fundamentals, data fetching, visualization
- [x] Week 2 Day 1 — Manual backtesting engine
- [x] Week 2 Day 2 — Transaction costs, Sharpe ratio
- [x] Week 2 Day 3 — Max drawdown + long-horizon validation
- [ ] Week 2 Day 4 — vectorbt parameter optimization
- [ ] Week 2 Day 5 — Multi-stock backtest report
- [ ] Week 3–4 — More indicators + data pipeline
- [ ] Phase 2 — Strategy development
- [ ] Phase 3 — ML/AI integration
- [ ] Phase 4 — Live trading + Agent

---

_Last updated: Week 2 Day 3 complete_
