import bitget.bitget_api as baseApi
from bitget.exceptions import BitgetAPIException
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import os

def fetch_historical_ohlcv(symbol, productType, granularity, endTime, limit):
    try:
        params={}
        params['symbol'] = symbol
        params['productType'] = productType
        params['granularity'] = granularity
        params['endTime'] = endTime
        params['limit'] = limit
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
    startTime = str(int(datetime(2024, 12, 1, 0, 0).timestamp())*1000)
    endTime = str(int(datetime(2025, 2, 1, 0, 0).timestamp())*1000)
    gran = '30m'

    load_dotenv()
    baseApi = baseApi.BitgetApi(os.environ.get('apiKey'), os.environ.get('secretKey'), os.environ.get('passphrase'))
    time = endTime
    data = []
    while startTime < time:
        data = fetch_historical_ohlcv('SBTCSUSDT', 'SUSDT-FUTURES', gran, time, '200') + data
        time = data[0][0]
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).astype(np.float32)
    df = add_index(df)
    df.to_csv(f'./dataset/dataset_{gran}.csv')