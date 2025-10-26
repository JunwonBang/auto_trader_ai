Cryptocurrency automatic future trading system using Bitget API and Reinforcement Learning (H1)

Preparation
1. Add .env file with 'api_key', 'secret_key', 'passphrase' which can be obtained from Bitget account.
2. Create virtual environment and install requirements.
   `python3 -m venv '<name of the virtual environment>'`
   `source <name of the virtual environment>/bin/activate` (for Mac)
   `pip install -r requirements.txt`

How the code works
1. create_dataset.py
   Get historical candlestick data(open, high, low, close, volume) of BTC-USDT Future with 1 minute granularity from Bitget.
   Add indices such as rsi, sma, ema, macd, macd signal.
   Save csv file to /dataset .
   
   Change 'start_time' and 'end_time' to adjust the period of dataset.
   Change params['granularity'] in get_historical_candlestick function to change granularity.

2. 
   
   
   
