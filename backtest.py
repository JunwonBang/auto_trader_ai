import bitget.bitget_api as baseApi
from bitget.exceptions import BitgetAPIException
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train import TradingEnv
from dotenv import load_dotenv
import os

def fetch_ohlcv(symbol, productType, granularity, limit):
    try:
        params={}
        params['symbol'] = symbol
        params['productType'] = productType
        params['granularity'] = granularity
        params['limit'] = limit
        data = baseApi.get('/api/v2/mix/market/candles', params)['data']
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_currency']).astype(np.float32)
        df.drop('volume_currency', axis=1, inplace=True)
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
    except BitgetAPIException as e:
        print("error:" + e.message)

if __name__ == '__main__':
    gran = '1H'
    total_timesteps = 10000

    load_dotenv()
    baseApi = baseApi.BitgetApi(os.environ.get('apiKey'), os.environ.get('secretKey'), os.environ.get('passphrase'))
    df = fetch_ohlcv('SBTCSUSDT', 'SUSDT-FUTURES', gran, '1000')
    env = TradingEnv(df)
    obs = env.reset()
    model = PPO.load(f"./models/ppo_trading_model_{gran}_{total_timesteps}")
    
    timesteps = []
    rewards = []
    timestep = 0
    total_reward = 0
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
        timesteps.append(timestep)
        rewards.append(total_reward)
        timestep += 1
        if done:
            break

    print("Backtest Total Reward:", total_reward)
    plt.figure(figsize=(10,5))
    plt.plot(timesteps, rewards, label='Cumulative Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Total Reward')
    plt.title('Backtest Reward over Time')
    plt.legend()
    plt.savefig(f'./backtest_results/cumulative_reward_{gran}_{total_timesteps}')