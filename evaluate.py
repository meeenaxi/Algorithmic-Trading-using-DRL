import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from stable_baselines3 import PPO
import os
import time

from config import *
from portfolio_env import PortfolioEnv
from model import CustomCNN 

def get_benchmark_data():
    """
    Robustly loads or downloads the benchmark data.
    """
    try:
        print(f"Loading benchmark data from file: {BENCHMARK_PATH}...")
        
        benchmark_data = pd.read_csv(
            BENCHMARK_PATH, 
            header=2, 
            index_col=0, 
            parse_dates=True
        )
    
        benchmark_data.index.name = 'Date'
        
        benchmark_data.sort_index(inplace=True)
        
        print("Benchmark loaded from cache successfully.")
        return benchmark_data

    except Exception as e:
        print(f"Warning: Could not load from cache ({e}).")
        print("Attempting to download Nifty 50 benchmark data directly...")
        
        print("Waiting 15 seconds before download...")
        time.sleep(15)
        
        try:
            benchmark_data = yf.download(BENCHMARK_TICKER, start=START_DATE, end=END_DATE)
            if benchmark_data.empty:
                raise Exception("No data returned from yfinance.")
                
            if 'Adj Close' in benchmark_data.columns:
                benchmark_prices = benchmark_data[['Adj Close']]
            else:
                benchmark_prices = benchmark_data[['Close']]
            
            benchmark_prices.to_csv(BENCHMARK_PATH)
            print(f"Benchmark data downloaded and saved to {BENCHMARK_PATH}")
            return benchmark_prices
            
        except Exception as download_e:
            print(f"\n[FATAL ERROR] Failed to download benchmark: {download_e}")
            print("Please wait 20-30 minutes for the rate limit to clear and try again.")
            return None

def run_clean_evaluation():
    print("--- Running Evaluation ---")
    
    print("Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH, index_col='Date', parse_dates=True)
    
    print("Loading the trained model...")
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
    )
    model = PPO.load(MODEL_PATH, policy_kwargs=policy_kwargs)
    
    price_cols = TICKERS
    signal_cols = [col for col in test_df.columns if col not in price_cols]
    
    print("Instantiating test environment...")
    test_steps = len(test_df) - WINDOW_LENGTH - 1
    test_env = PortfolioEnv(
        data_df=test_df, 
        ticker_list=price_cols, 
        signal_list=signal_cols,
        window_length=WINDOW_LENGTH,
        start_date_index = WINDOW_LENGTH - 1, 
        steps = test_steps
    )

    print("Running evaluation backtest...")
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated

    results_df = pd.DataFrame(test_env.info_list).set_index('date')
    
    benchmark_prices = get_benchmark_data()
    
    if benchmark_prices is None:
        print("Could not get benchmark data. Exiting evaluation.")
        return

    benchmark_test = benchmark_prices.reindex(test_df.index, method='ffill')
    benchmark_test.bfill(inplace=True) 

    benchmark_returns = benchmark_test.pct_change().iloc[:, 0].fillna(0)
    benchmark_value = (1 + benchmark_returns).cumprod()

    start_date = results_df.index[0]
    end_date = results_df.index[-1]
    benchmark_to_plot = benchmark_value.loc[start_date:end_date]
    
    benchmark_to_plot = benchmark_to_plot / benchmark_to_plot.iloc[0]
    
    results_df['benchmark_value'] = benchmark_to_plot.values
    
    RESULTS_PATH = f'{MODEL_OUTPUT_DIR}/evaluation_results.csv'
    results_df.to_csv(RESULTS_PATH)
    print(f"Evaluation complete. Results saved to {RESULTS_PATH}")
    
    plt.figure(figsize=(15, 6))
    plt.plot(results_df['portfolio_value'], label='DRL Agent', color='blue')
    plt.plot(results_df['benchmark_value'], label='Nifty 50 Index (^NSEI)', color='orange', linestyle='--')
    plt.title('Agent Performance vs. Nifty 50 Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Normalized)')
    plt.legend()
    plt.grid(True)
    
    plot_path = f'{MODEL_OUTPUT_DIR}/final_evaluation_plot.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_clean_evaluation()