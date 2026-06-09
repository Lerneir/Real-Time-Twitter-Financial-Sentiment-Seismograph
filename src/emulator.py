import time
import json
import random
import os
import pandas as pd

def emulate_stream(csv_path: str, output_dir: str, chunk_size: int = 50, interval: int = 10, stop_event = None):
    """
    Simulates a live Twitter stream by chunking a source CSV dataset and
    periodically saving batches as JSON files to an active ingestion landing directory.
    """
    print(f"[EMULATOR] Starting chunker. Source: {csv_path} | Output: {output_dir}")
    
    # Path resolution for local vs DBFS context
    real_csv_path = csv_path.replace("dbfs:", "/dbfs") if csv_path.startswith("dbfs:") else csv_path
    real_output_dir = output_dir.replace("dbfs:", "/dbfs") if output_dir.startswith("dbfs:") else output_dir
    
    os.makedirs(real_output_dir, exist_ok=True)
    
    # Load source CSV
    if not os.path.exists(real_csv_path):
        print(f"[EMULATOR] Warning: CSV path '{real_csv_path}' not found. Generating dummy dataset...")
        dummy_tweets = [
            "Tesla stock is soaring to the moon! Incredible gains today. $TSLA",
            "Apple reports record-breaking quarterly earnings. $AAPL",
            "Huge market selloff incoming. Inflation rates are higher than expected.",
            "Bitcoin crashes below support level. Panic selling everywhere! $BTC",
            "NVIDIA launches new AI chips, market responds bullishly. $NVDA",
            "Amazon faces supply chain bottlenecks, stock price drops. $AMZN",
            "Federal Reserve hints at interest rate cuts, markets rally.",
            "Short sellers targeting tech sector, major volatility expected.",
            "Microsoft acquires leading gaming studio, bullish signal. $MSFT",
            "Crude oil prices tumble, energy stocks under pressure."
        ]
        df = pd.DataFrame({
            "text": dummy_tweets * 50,
            "label": [random.choice([0, 1, 2]) for _ in range(500)]
        })
    else:
        df = pd.read_csv(real_csv_path)
        
    print(f"[EMULATOR] Data source loaded with {len(df)} rows. Commencing emulation...")
    
    chunk_idx = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            print("[EMULATOR] Stop event detected. Shutting down emulator thread...")
            break
            
        start_row = (chunk_idx * chunk_size) % len(df)
        end_row = start_row + chunk_size
        chunk = df.iloc[start_row:end_row]
        
        records = []
        for _, row in chunk.iterrows():
            records.append({
                "text": str(row["text"]),
                "label": int(row["label"]) if "label" in row else 2,
                "timestamp": int(time.time()),
                "followers": random.randint(100, 500000)
            })
            
        batch_filename = f"batch_{chunk_idx}_{int(time.time())}.json"
        batch_file_path = os.path.join(real_output_dir, batch_filename)
        
        with open(batch_file_path, "w") as f:
            json.dump(records, f)
            
        print(f"[EMULATOR] Ingested chunk {chunk_idx + 1} to {batch_file_path}")
        chunk_idx += 1
        
        # Check stop_event or sleep for the interval in smaller steps
        sleep_left = interval
        while sleep_left > 0:
            if stop_event is not None and stop_event.is_set():
                break
            sleep_step = min(1, sleep_left)
            time.sleep(sleep_step)
            sleep_left -= sleep_step
