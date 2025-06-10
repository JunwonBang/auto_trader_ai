import gym
from stable_baselines3 import PPO
import pandas as pd
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        
        # Action space: 0=HOLD, 1=OPEN LONG, 2=OPEN SHORT, 3=CLOSE
        self.action_space = gym.spaces.Discrete(4)
        
        # Observation space: price data + technical indicators + portfolio state
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(16,),  # 11 market features + 5 portfolio state features
            dtype=np.float32
        )

        # Trading parameters
        self.quantity = 0.001
        self.trading_fee = 0.0006
        self.leverage = 10
        self.max_position_hold_time = 100  # Maximum number of steps to hold a position
        self.position_hold_time = 0
        
        # Portfolio state
        self.current_position = None
        self.entry_price = None
        self.liquidation_price = None
        self.portfolio_value = 1000  # Initial portfolio value
        self.max_portfolio_value = 1000  # Track maximum portfolio value
        self.total_trades = 0
        self.winning_trades = 0
        self.unrealized_pnl = 0

    def reset(self):
        self.current_step = 0
        self.current_position = None
        self.entry_price = None
        self.liquidation_price = None
        self.portfolio_value = 1000
        self.max_portfolio_value = 1000
        self.position_hold_time = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.unrealized_pnl = 0
        return self._next_observation()
    
    def _next_observation(self):
        # Get market data
        obs = self.df.iloc[self.current_step][
            ['open', 'high', 'low', 'close', 'volume', 
             'rsi', 'sma', 'ema_12', 'ema_26', 'macd', 'macd_signal']
        ].values
        
        # Add portfolio state
        portfolio_state = np.array([
            self.portfolio_value / self.max_portfolio_value,  # Normalized portfolio value
            1 if self.current_position == "long" else 0,     # Long position flag
            1 if self.current_position == "short" else 0,    # Short position flag
            self.position_hold_time / self.max_position_hold_time,  # Normalized position hold time
            self.unrealized_pnl / self.portfolio_value if self.portfolio_value != 0 else 0  # Normalized unrealized PnL
        ])
        
        return np.concatenate([obs, portfolio_state]).astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        current_price = self.df.iloc[self.current_step]['close']
        prev_portfolio_value = self.portfolio_value
        reward = 0
        self.unrealized_pnl = 0
        
        # Calculate unrealized PnL for current position
        if self.current_position == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity * self.leverage
        elif self.current_position == "short":
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity * self.leverage
        else:
            self.unrealized_pnl = 0
            
        # Position holding time penalty
        if self.current_position is not None:
            self.position_hold_time += 1
            if self.position_hold_time > self.max_position_hold_time:
                reward -= 0.1  # Penalty for holding too long
        
        # Trading action rewards
        if action == 0:  # HOLD
            reward += self.unrealized_pnl * 0.1  # Small reward for holding profitable position
        elif action == 1:  # OPEN LONG
            if self.current_position is None:
                fee = current_price * self.quantity * self.leverage * self.trading_fee
                self.entry_price = current_price
                self.liquidation_price = self.entry_price * (1 - self.leverage/100)
                self.current_position = "long"
                self.position_hold_time = 0
                self.portfolio_value -= fee
        elif action == 2:  # OPEN SHORT
            if self.current_position is None:
                fee = current_price * self.quantity * self.leverage * self.trading_fee
                self.entry_price = current_price
                self.liquidation_price = self.entry_price * (1 + self.leverage/100)
                self.current_position = "short"
                self.position_hold_time = 0
                self.portfolio_value -= fee
        elif action == 3:  # CLOSE
            if self.current_position is not None:
                if self.current_position == "long":
                    profit = (current_price - self.entry_price) * self.quantity * self.leverage
                else:  # short
                    profit = (self.entry_price - current_price) * self.quantity * self.leverage
                    
                fee = current_price * self.quantity * self.leverage * self.trading_fee
                self.portfolio_value += profit - fee
                
                # Update portfolio statistics
                self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
                self.total_trades += 1
                if profit > fee:
                    self.winning_trades += 1
                
                # Reset position
                self.current_position = None
                self.entry_price = None
                self.liquidation_price = None
                self.position_hold_time = 0

        # Handle liquidation
        if self.current_position == "long" and current_price <= self.liquidation_price:
            loss = -self.entry_price * self.quantity
            self.portfolio_value += loss
            self.current_position = None
            self.entry_price = None
            self.liquidation_price = None
            self.position_hold_time = 0
        
        if self.current_position == "short" and current_price >= self.liquidation_price:
            loss = -self.entry_price * self.quantity
            self.portfolio_value += loss
            self.current_position = None
            self.entry_price = None
            self.liquidation_price = None
            self.position_hold_time = 0

        # 보상: 포트폴리오 가치 변화량
        reward += self.portfolio_value - prev_portfolio_value

        # Add portfolio value to info
        info = {
            'portfolio_value': self.portfolio_value,
            'max_portfolio_value': self.max_portfolio_value,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades)
        }

        return self._next_observation(), reward, done, info

if __name__ == '__main__':
    df = pd.read_csv(f'./dataset/data_20250401_20250501.csv')
    env = TradingEnv(df)
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
    model.learn(total_timesteps=100000)
    model.save(f"./models/ppo2")