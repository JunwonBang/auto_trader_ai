# Cryptocurrency automatic future trading system using Bitget API and Reinforcement Learning

## Preparation
- ### Create a `.env` file in the project root directory and add your Bitget API credentials.
e.g.,
```
api_key='your_api_key'
secret_key='your_secret_key'
passphrase='your_passphrase'
```
- ### Create virtual environment & Install dependencies.

`python3 -m venv <name of the virtual environment>`

`source <name of the virtual environment>/bin/activate `(for Mac)

`pip install -r requirements.txt`

## How It Works
### 1. create_dataset.py
- Fetches historical candlestick data (open, high, low, close, volume) for BTC-USDT Futures
from the Bitget API with 1-minute granularity
- Calculates key technical indicators:
    - RSI (Relative Strength Index)
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - MACD (Moving Average Convergence Divergence)
    - MACD Signal
- Save the dataset as a `.csv` file under `/dataset`.

#### Adjustments
- Change start_time and end_time to modify the data period.
- Modify `params['granularity']` in `get_historical_candlestick()` to change time intervals.

### 2. train.py
- Load dataset from `/dataset`
- Train a reinforcement learning model
- Save the trained model to `/models`

### 3. backtest.py
- Load the trained model from `/models`
- Run backtesting
- Save results to `/backtest_results`

### 4. trade.py
- Load the trained model from `/models`
- Run live trading

## Example Workflow
Step 1: `python create_dataset.py`

Step 2: `python train.py`

Step 3: `python backtest.py`

Step 4: `python trade.py`

## Author
Jude (Junwon) Bang
jude.bang@mail.utoronto.ca