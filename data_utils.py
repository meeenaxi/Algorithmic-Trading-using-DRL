# data_utils.py
# NEW VERSION: Includes batch-downloading to avoid rate limits.

import pandas as pd
import yfinance as yf
import os
import technical_indicators as ti
from sklearn.preprocessing import StandardScaler
from config import * # Import all our settings
import time
import numpy as np

def download_stock_data():
    """
    Downloads and caches the 50 stock tickers IN BATCHES to avoid rate limits.
    """
    if os.path.exists(RAW_DATA_PATH):
        print(f"Loading stock data from local cache: {RAW_DATA_PATH}")
        return pd.read_csv(RAW_DATA_PATH, index_col='Date', parse_dates=True)
    
    print("Starting batched download for all 50 tickers...")
    
    all_chunks = []
    batch_size = 10  # 10 tickers per batch
    num_batches = int(np.ceil(len(TICKERS) / batch_size))
    ticker_batches = np.array_split(TICKERS, num_batches)
    
    for i, batch in enumerate(ticker_batches):
        try:
            print(f"  Downloading batch {i + 1}/{num_batches} ({len(batch)} tickers)...")
            batch_tickers = list(batch) # Convert numpy array to list
            
            # Download the current batch
            raw_data = yf.download(batch_tickers, start=START_DATE, end=END_DATE)
            
            if raw_data.empty:
                raise Exception("No data returned for this batch")
            
            # Get the correct price column ('Adj Close' or 'Close')
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                data_chunk = raw_data['Adj Close']
            else:
                data_chunk = raw_data['Close']
            
            # For batches of 1, yf returns a simple DataFrame. We need to select the tickers.
            if len(batch_tickers) == 1:
                data_chunk = data_chunk[batch_tickers]
            
            all_chunks.append(data_chunk)
            
            if i < num_batches - 1: # Don't wait after the last batch
                print(f"  ...Success. Waiting 30 seconds to avoid rate limit...")
                time.sleep(30) # Wait 30 seconds
                
        except Exception as e:
            print(f"  --- FAILED to download batch {i + 1}: {e}")
            print(f"  --- Skipping tickers: {list(batch)}")

    if not all_chunks:
        print("\nERROR: No stock data could be downloaded. Exiting.")
        return pd.DataFrame()

    # Combine all downloaded chunks
    all_data = pd.concat(all_chunks, axis=1)
    
    # Clean up any potential failed columns that might be all NaN
    all_data.dropna(axis=1, how='all', inplace=True)
    
    all_data.to_csv(RAW_DATA_PATH)
    print(f"\nStock data saved successfully to '{RAW_DATA_PATH}'")
    return all_data

def download_benchmark_data():
    """
    Downloads and caches the Nifty 50 Index data.
    Waits 30 seconds *before* downloading to be safe.
    """
    if os.path.exists(BENCHMARK_PATH):
        print(f"Loading benchmark data from local cache: {BENCHMARK_PATH}")
        return pd.read_csv(BENCHMARK_PATH, index_col='Date', parse_dates=True)
    
    print("Waiting 30 seconds before downloading benchmark to be safe...")
    time.sleep(30) # Wait 30 seconds before this new request
    
    print(f"Downloading benchmark ticker {BENCHMARK_TICKER}...")
    try:
        data = yf.download(BENCHMARK_TICKER, start=START_DATE, end=END_DATE)
        if data.empty: raise Exception("No data returned for benchmark.")
        
        if 'Adj Close' in data.columns:
            benchmark_data = data[['Adj Close']]
        else:
            benchmark_data = data[['Close']]
            
        benchmark_data.to_csv(BENCHMARK_PATH)
        print(f"Benchmark data saved successfully to '{BENCHMARK_PATH}'")
        return benchmark_data
        
    except Exception as e:
        print(f"ERROR downloading benchmark data: {e}")
        return pd.DataFrame()

def create_features(all_data):
    """Creates features for the 50 stocks."""
    print("Calculating technical indicators...")
    signals_df = pd.DataFrame(index=all_data.index)
    
    # This might take a moment for 50 stocks
    for ticker in all_data.columns:
        signals_df[f'RSI_{ticker}'] = ti.calculate_rsi(all_data[ticker])
        signals_df[f'SMA_{ticker}'] = ti.calculate_sma(all_data[ticker])
        upper, middle, lower = ti.calculate_bbands(all_data[ticker])
        signals_df[f'BB_Upper_{ticker}'] = upper
        signals_df[f'BB_Middle_{ticker}'] = middle
        signals_df[f'BB_Lower_{ticker}'] = lower
    
    print(f"Engineered {signals_df.shape[1]} total features.")
    return signals_df

def process_and_save_data():
    """Main function to run the full data pipeline for the stocks."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Step 1: Download stock data (using new patient batching)
    all_data = download_stock_data()
    if all_data.empty: return

    # Step 2: Create features
    signals_df = create_features(all_data)
    
    # Step 3: Combine, clean, and split
    final_df = all_data.join(signals_df)
    final_df.dropna(inplace=True)
    
    train_size = int(len(final_df) * TRAIN_TEST_SPLIT)
    train_df = final_df.iloc[:train_size].copy()
    test_df = final_df.iloc[train_size:].copy()
    
    # Step 4: Scale features
    # This will take a moment, it's a lot of data
    print("Scaling features... This may take a moment.")
    price_cols = [col for col in all_data.columns if col in TICKERS]
    signal_cols = [col for col in signals_df.columns]
    
    scaler = StandardScaler()
    scaler.fit(train_df[signal_cols])
    
    train_df[signal_cols] = scaler.transform(train_df[signal_cols])
    test_df[signal_cols] = scaler.transform(test_df[signal_cols])
    
    # Step 5: Save final processed files
    train_df.to_csv(TRAIN_DATA_PATH)
    test_df.to_csv(TEST_DATA_PATH)
    
    print(f"Processed stock data saved to {TRAIN_DATA_PATH} and {TEST_DATA_PATH}")

if __name__ == "__main__":
    process_and_save_data()  # Runs the main pipeline for stocks
    download_benchmark_data()  # Also downloads the benchmark data