import bitget.bitget_api as baseApi
from bitget.exceptions import BitgetAPIException

import gym
from stable_baselines3 import PPO

import pandas as pd
import numpy as np

from dotenv import load_dotenv
import os

from datetime import datetime

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

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11, ), dtype=np.float32)

        self.quantity = 0.001
        self.trading_fee = 0.0006
        self.leverage = 10
        self.current_position = None
        self.entry_price = None
        self.liquidation_price = None

    def reset(self):
        self.current_step = 0
        self.current_position = None
        self.entry_price = None
        self.liquidation_price = None
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.df.iloc[self.current_step][['open', 'high', 'low', 'close', 'volume', 'rsi', 'sma', 'ema_12', 'ema_26', 'macd', 'macd_signal']].values
        # obs = self.df.iloc[self.current_step-100:self.current_step][['open', 'high', 'low', 'close', 'volume']].values
        return obs.astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        
        if action == 0: # HOLD
            pass
        elif action == 1: # OPEN LONG
            if self.current_position is None:
                self.entry_price = current_price
                self.liquidation_price = self.entry_price * (1 - self.leverage/100)
                fee = self.entry_price*self.quantity*self.leverage*self.trading_fee
                reward = -fee
                self.current_position = "long"
        elif action == 2: # OPEN SHORT
            if self.current_position is None:
                self.entry_price = current_price
                self.liquidation_price = self.entry_price * (1 + self.leverage/100)
                fee = self.entry_price*self.quantity*self.leverage*self.trading_fee
                reward = -fee
                self.current_position = "short"
        elif action == 3: # CLOSE
            if self.current_position == "long":
                profit = (current_price - self.entry_price)*self.quantity*self.leverage
                fee = current_price*self.quantity*self.leverage*self.trading_fee
                reward = profit-fee
            elif self.current_position == "short":
                profit = (self.entry_price - current_price)*self.quantity*self.leverage
                fee = current_price*self.quantity*self.leverage*self.trading_fee
                reward = profit-fee
            self.current_position = None
            self.entry_price = None
            self.liquidation_price = None

        # MARGIN CALL
        if self.current_position == "long" and current_price <= self.liquidation_price:
            reward = -self.entry_price*self.quantity
            self.current_position = None
            self.entry_price = None
            self.liquidation_price = None
        
        if self.current_position == "short" and current_price >= self.liquidation_price:
            reward = -self.entry_price*self.quantity
            self.current_position = None
            self.entry_price = None
            self.liquidation_price = None

        return self._next_observation(), reward, done, {}

if __name__ == '__main__':
    # load_dotenv()
    # baseApi = baseApi.BitgetApi(os.environ.get('apiKey'), os.environ.get('secretKey'), os.environ.get('passphrase'))
    # startTime = str(int(datetime(2024, 12, 1, 0, 0).timestamp())*1000)
    # endTime = str(int(datetime(2025, 2, 1, 0, 0).timestamp())*1000)
    # time = endTime
    # data = []
    # while startTime < time:
    #     data = fetch_historical_ohlcv('SBTCSUSDT', 'SUSDT-FUTURES', '1m', time, '200') + data
    #     time = data[0][0]
    # df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).astype(np.float32)
    # df = add_index(df)
    # df.to_csv('./dataset/dataset.csv')
    df = pd.read_csv('./dataset/dataset.csv')

    env = TradingEnv(df)

    total_timesteps = 10000
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.00025,
        n_steps=4096,
        batch_size=1024,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(f"./models/ppo_trading_model_{total_timesteps}")
