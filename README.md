# Cryptocurrency automatic future trading system using Bitget API and Reinforcement Learning

## Preparation
- ### Create a '.env' file in the project root directory and add your Bitget API credentials.
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

## How the code works
### 1. create_dataset.py

Get historical candlestick data(open, high, low, close, volume) of BTC-USDT Future with 1 minute granularity from Bitget.

Add indices such as rsi, sma, ema, macd, macd signal.

Save csv file to /dataset .
   

Change 'start_time' and 'end_time' to adjust the period of dataset.

Change params['granularity'] in get_historical_candlestick function to change granularity.

### 2. train.py
Load dataset from /dataset

Save trained model to /models

### 3. backtest.py
Save results to /backtest_results

### 4. trade.py
   
   
   
