import bitget.bitget_api as baseApi
from bitget.exceptions import BitgetAPIException
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train import TradingEnv
from datetime import datetime
import os

def run_backtest(env, model):
    obs = env.reset()
    timestep = 0
    timesteps = []
    total_reward = 0
    rewards = []
    
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
        timesteps.append(timestep)
        rewards.append(total_reward)
        timestep += 1
        if done:
            break
            
    return timesteps, rewards

if __name__ == '__main__':
    df = pd.read_csv(f'./dataset/data_20250401_20250501.csv')
    env = TradingEnv(df)
    model = PPO.load(f"./models/ppo")
    
    n_runs = 10
    all_timesteps = []
    all_rewards = []
    
    for i in range(n_runs):
        print(f"Running backtest {i+1}/{n_runs}")
        timesteps, rewards= run_backtest(env, model)
        all_timesteps.append(timesteps)
        all_rewards.append(rewards)
    
    plt.figure(figsize=(12, 6))
    
    for i in range(n_runs):
        plt.plot(all_timesteps[i], all_rewards[i], alpha=0.3, label=f'Run {i+1}' if i < 3 else None)
    
    # Plot mean reward
    mean_cumulative_rewards = np.mean(all_rewards, axis=0)
    plt.plot(all_timesteps[0], mean_cumulative_rewards, 'k--', linewidth=2, label='Mean')
    
    # Add confidence interval
    std_cumulative_rewards = np.std(all_rewards, axis=0)
    plt.fill_between(all_timesteps[0], 
                     mean_cumulative_rewards - std_cumulative_rewards,
                     mean_cumulative_rewards + std_cumulative_rewards,
                     alpha=0.2, color='gray', label='±1 std')
    
    plt.xlabel('Timesteps')
    plt.ylabel('Cumulative Rewards')
    plt.legend()
    
    plt.savefig(f'./backtest_results/cumulative_rewards.png')
    plt.close()