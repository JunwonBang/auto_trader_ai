import gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df, window_size=10):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.action_space = gym.spaces.Discrete(4)  # 0: HOLD, 1: LONG, 2: SHORT, 3: CLOSE
        # 관찰값: 최근 window_size개의 close, volume, rsi, ... + 포트폴리오 상태 4개
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size*6 + 4,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.cash = 1000
        self.shares = 0
        self.total_asset = 1000
        self.max_asset = 1000
        return self._get_obs()

    def _get_obs(self):
        # 최근 window_size개의 데이터 + 포트폴리오 상태
        window = self.df.iloc[self.current_step-self.window_size:self.current_step]
        obs = np.concatenate([
            window[['close', 'volume', 'rsi', 'sma', 'ema_12', 'ema_26']].values.flatten(),
            np.array([
                self.cash,
                self.shares,
                self.position,
                self.total_asset
            ])
        ])
        return obs.astype(np.float32)

    def step(self, action):
        done = False
        price = self.df.iloc[self.current_step]['close']
        prev_asset = self.total_asset

        # 거래 로직
        if action == 1 and self.position == 0:  # LONG 진입
            self.position = 1
            self.entry_price = price
            self.shares = self.cash / price
            self.cash = 0
        elif action == 2 and self.position == 0:  # SHORT 진입
            self.position = -1
            self.entry_price = price
            self.shares = self.cash / price
            self.cash = 0
        elif action == 3 and self.position != 0:  # 포지션 청산
            if self.position == 1:
                self.cash = self.shares * price
            elif self.position == -1:
                self.cash = self.shares * (2 * self.entry_price - price)
            self.position = 0
            self.shares = 0
            self.entry_price = 0

        # 자산 계산
        if self.position == 1:
            self.total_asset = self.shares * price
        elif self.position == -1:
            self.total_asset = self.shares * (2 * self.entry_price - price)
        else:
            self.total_asset = self.cash

        self.max_asset = max(self.max_asset, self.total_asset)

        # 보상: log-return + 거래 패널티
        reward = np.log(self.total_asset / prev_asset + 1e-8)
        if action in [1, 2, 3]:
            reward -= 0.001  # 거래 패널티

        self.current_step += 1
        if self.current_step >= len(self.df) - 1 or self.total_asset <= 0:
            done = True

        return self._get_obs(), reward, done, {
            'total_asset': self.total_asset,
            'max_asset': self.max_asset
        }
