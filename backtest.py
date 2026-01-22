import bitget.bitget_api as baseApi
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train import TradingEnv
from matplotlib.ticker import MaxNLocator

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate the annualized Sharpe ratio of a returns stream."""
    excess_returns = returns - risk_free_rate/252  # Assuming 252 trading days in a year
    return np.sqrt(252) * np.mean(excess_returns) / (np.std(excess_returns) + 1e-9)

def calculate_max_drawdown(cum_returns):
    """Calculate the maximum drawdown of a cumulative returns series."""
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / (peak + 1e-9)  # Add small number to avoid division by zero
    return np.min(drawdown)

def run_backtest(env, model):
    obs = env.reset()
    timestep = 0
    timesteps = []
    total_reward = 0
    rewards = []
    portfolio_values = [1.0]  # Start with portfolio value of 1.0
    
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Track portfolio value (assuming reward is the return at each step)
        portfolio_values.append(portfolio_values[-1] * (1 + reward))
        
        timesteps.append(timestep)
        rewards.append(total_reward)
        timestep += 1
        if done:
            break
    
    # Calculate returns from portfolio values
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate metrics
    sharpe_ratio = calculate_sharpe_ratio(returns)
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    return timesteps, rewards, portfolio_values[1:], sharpe_ratio, max_drawdown

if __name__ == '__main__':
    import os
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    
    # Create backtest results directory if it doesn't exist
    os.makedirs('./backtest_results', exist_ok=True)
    
    df = pd.read_csv('./dataset/data_20250501_20250601.csv')
    env = TradingEnv(df)
    model = PPO.load("./models/ppo")
    
    n_runs = 10
    all_timesteps = []
    all_rewards = []
    all_portfolio_values = []
    all_sharpe_ratios = []
    all_max_drawdowns = []
    
    print("\n=== Running Backtests ===")
    for i in range(n_runs):
        print(f"Backtest {i+1}/{n_runs}...")
        timesteps, rewards, portfolio_values, sharpe_ratio, max_drawdown = run_backtest(env, model)
        
        all_timesteps.append(timesteps)
        all_rewards.append(rewards)
        all_portfolio_values.append(portfolio_values)
        all_sharpe_ratios.append(sharpe_ratio)
        all_max_drawdowns.append(max_drawdown)
        
        print(f"  Run {i+1} - Sharpe Ratio: {sharpe_ratio:.4f}, Max Drawdown: {max_drawdown*100:.2f}%")
    
    # Calculate mean metrics
    mean_sharpe = np.mean(all_sharpe_ratios)
    mean_max_drawdown = np.mean(all_max_drawdowns)
    mean_cumulative_return = (np.array([pv[-1] for pv in all_portfolio_values]) - 1).mean() * 100
    
    print("\n=== Backtest Results ===")
    print(f"Mean Sharpe Ratio: {mean_sharpe:.4f}")
    print(f"Mean Max Drawdown: {mean_max_drawdown*100:.2f}%")
    print(f"Mean Cumulative Return: {mean_cumulative_return:.2f}%")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Plot cumulative returns
    for i in range(n_runs):
        ax1.plot(all_timesteps[i], all_portfolio_values[i], alpha=0.3, label=f'Run {i+1}' if i < 3 else None)
    
    # Plot mean portfolio value
    mean_portfolio_values = np.mean(all_portfolio_values, axis=0)
    ax1.plot(all_timesteps[0], mean_portfolio_values, 'k--', linewidth=2, label='Mean')
    
    # Add confidence interval
    std_portfolio_values = np.std(all_portfolio_values, axis=0)
    ax1.fill_between(all_timesteps[0], 
                    mean_portfolio_values - std_portfolio_values,
                    mean_portfolio_values + std_portfolio_values,
                    alpha=0.2, color='gray', label='±1 std')
    
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Portfolio Value (Multiple of Initial)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text box with metrics
    metrics_text = (f'Mean Sharpe Ratio: {mean_sharpe:.2f}\n'
                   f'Mean Max Drawdown: {mean_max_drawdown*100:.2f}%\n'
                   f'Mean Return: {mean_cumulative_return:.2f}%')
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot drawdowns
    for i in range(n_runs):
        portfolio_values = all_portfolio_values[i]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / (peak + 1e-9)
        ax2.fill_between(all_timesteps[i], drawdown * 100, 0, alpha=0.3, label=f'Run {i+1}' if i < 3 else None)
    
    ax2.set_title('Drawdowns')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('./backtest_results/backtest_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nBacktest completed. Results saved to backtest_results/backtest_results.png")