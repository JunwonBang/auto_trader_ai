import bitget.bitget_api as baseApi
from bitget.exceptions import BitgetAPIException
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import os

def get_historical_candlestick(symbol, productType, endTime):
    try:
        params={}
        params['symbol'] = symbol
        params['productType'] = productType
        params['granularity'] = '1m'
        params['endTime'] = endTime
        params['limit'] = '200'
        data = baseApi.get('/api/v2/mix/market/history-candles', params)['data']
        for i in range(len(data)):
            del data[i][-1]
        return data
    except BitgetAPIException as e:
        print("error:" + e.message)

def add_index(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=20).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=20).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['sma'] = df['close'].rolling(window=14).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df = df.replace(' ', pd.NA)
    df = df.dropna()
    return df

if __name__ == '__main__':
    start_time = str(int(datetime(2025, 4, 1).timestamp()*1000))
    end_time = str(int(datetime(2025, 5, 1).timestamp()*1000))

    load_dotenv()
    baseApi = baseApi.BitgetApi(os.environ.get('api_key'), os.environ.get('secret_key'), os.environ.get('passphrase'))
    time = end_time
    data = []
    while time > start_time:
        data = get_historical_candlestick('BTCUSDT', 'USDT-FUTURES', time) + data
        time = data[0][0]
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).astype(np.float32)
    df = add_index(df)
    df.to_csv(f'./dataset/data.csv')