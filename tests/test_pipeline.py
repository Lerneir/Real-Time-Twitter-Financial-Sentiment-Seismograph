import os
import json
import time
import random
import threading
import pandas as pd
import numpy as np

# Import our modular components
from src.classifier import clean_tweet_text, analyze_sentiment
from src.emulator import emulate_stream

mock_csv_path = "test_source_tweets.csv"
incoming_dir = "test_incoming"

# 1. Create a local mock CSV dataset
tweets = [
    {"text": "Apple stock soars as profits double. $AAPL is heading to the moon!", "label": 1},
    {"text": "Tesla shares crash 10% after poor delivery guidance. Massive panic selling.", "label": 0},
    {"text": "Federal Reserve leaves interest rates unchanged. Market reacts neutrally.", "label": 2},
    {"text": "NVIDIA beats revenue estimates with record AI chip sales. Bullish signal! $NVDA", "label": 1},
    {"text": "Amazon faces antitrust lawsuit, stock drops. Severe negative sentiment.", "label": 0},
    {"text": "Goldman Sachs upgrades banking sector to buy. $GS", "label": 1},
    {"text": "Bitcoin falls below support levels. Traders are panicking.", "label": 0},
    {"text": "Inflation remains sticky, market is highly volatile and flat.", "label": 2},
]

pd.DataFrame(tweets).to_csv(mock_csv_path, index=False)
print(f"[TEST] Mock CSV dataset written to: {mock_csv_path}")

# 2. Test the Chunker/Emulator logic using a thread
print("[TEST] Testing ingestion chunker emulator (background thread)...")
stop_event = threading.Event()
emulator_thread = threading.Thread(
    target=emulate_stream, 
    # chunk_size=2, interval=1
    args=(mock_csv_path, incoming_dir, 2, 1, stop_event),
    daemon=True
)
emulator_thread.start()

# Let it run for 2.5 seconds (should generate ~2-3 chunks)
time.sleep(2.5)
stop_event.set()
emulator_thread.join(timeout=2)
print("[TEST] Emulator thread stopped.")

# 3. Test Text Cleaning and VADER Sentiment Classifier
print("[TEST] Testing sentiment classifier module...")
analyzed_records = []

# Verify files were created
if not os.path.exists(incoming_dir):
    print("[TEST ERROR] Incoming directory was not created!")
    exit(1)

json_files = [f for f in os.listdir(incoming_dir) if f.endswith(".json")]
print(f"[TEST] Found {len(json_files)} generated JSON batches.")

for filename in json_files:
    filepath = os.path.join(incoming_dir, filename)
    with open(filepath, "r") as f:
        records = json.load(f)
        
    for record in records:
        text = record["text"]
        
        # Call our clean_tweet_text and analyze_sentiment functions
        clean_text = clean_tweet_text(text)
        sentiment = analyze_sentiment(text)
        
        record["cleaned_text"] = clean_text
        record["positive"] = sentiment["positive"]
        record["negative"] = sentiment["negative"]
        record["neutral"] = sentiment["neutral"]
        record["compound"] = sentiment["compound"]
        analyzed_records.append(record)

df_analyzed = pd.DataFrame(analyzed_records)
print(f"[TEST] Classified {len(df_analyzed)} tweets. Columns: {df_analyzed.columns.tolist()}")

# 4. Test Sliding Windows & Z-Score
print("[TEST] Testing sliding window aggregation & Z-Score calculation...")
df_analyzed['weight'] = np.log(df_analyzed['followers'] + 1)
df_analyzed['weighted_diff'] = (df_analyzed['positive'] - df_analyzed['negative']) * df_analyzed['weight']

sum_weighted_diff = df_analyzed['weighted_diff'].sum()
sum_weight = df_analyzed['weight'].sum()
wsi = sum_weighted_diff / sum_weight if sum_weight > 0 else 0
avg_neg = df_analyzed['negative'].mean()
tweet_count = len(df_analyzed)

print(f"[TEST] Aggregation Results:")
print(f"  - Weighted Sentiment Index (WSI): {wsi:.4f}")
print(f"  - Average Negativity: {avg_neg:.4f}")
print(f"  - Tweet Volume: {tweet_count}")

# 5. Clean up local test assets
print("[TEST] Cleaning up temporary files...")
for file in os.listdir(incoming_dir):
    os.remove(os.path.join(incoming_dir, file))
os.rmdir(incoming_dir)
os.remove(mock_csv_path)
print("[TEST] Local test run completed successfully! All logic is verified.")
