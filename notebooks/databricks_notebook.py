# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# dependencies = [
#   "nltk",
#   "pandas",
#   "requests",
#   "gitpython",
# ]
# ///
# MAGIC %md
# MAGIC # Real-Time Twitter Financial Sentiment Seismograph
# MAGIC **Course Module Integration:**
# MAGIC 1. **Mining Data Streams:** PySpark Structured Streaming processes live-emulated JSON tweet chunks.
# MAGIC 2. **Natural Language Processing (NLP):** Spark SQL regular expressions perform raw tweet cleaning.
# MAGIC 3. **Sentiment Analysis:** An optimized NLTK VADER classifier runs inside a Pandas UDF.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1: Install Dependencies & Setup Configurations
# MAGIC Installs the NLTK library, downloads the VADER lexicon, configures Databricks Widgets for runtime credentials, and sets workspace paths for the ingestion landing zone, Spark checkpoints, and CSV output.

# COMMAND ----------

# MAGIC %pip install nltk

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import nltk
nltk.download('vader_lexicon', quiet=True)

# Safeguard for running outside of Databricks (e.g. in local Jupyter notebooks)
try:
    dbutils
except NameError:
    class MockWidgets:
        def text(self, name, default_value, label=""):
            pass
        def get(self, name):
            import os
            return os.getenv(name.upper(), "")
    class MockDBUtils:
        widgets = MockWidgets()
    dbutils = MockDBUtils()

# Define Widgets for Github & Kaggle Credentials & Paths
dbutils.widgets.text("github_token", "", "GitHub Personal Access Token")
dbutils.widgets.text("github_repo", "Lerneir/Real-Time-Twitter-Financial-Sentiment-Seismograph", "GitHub Repository (owner/repo)")
dbutils.widgets.text("github_branch", "main", "GitHub Branch")
dbutils.widgets.text("github_file_path", "aggregated_metrics.csv", "GitHub File Path")
dbutils.widgets.text("github_tweets_file_path", "important_tweets.csv", "GitHub Tweets File Path")
dbutils.widgets.text("csv_source_path", "/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/data/train_data.csv", "Source CSV Path (DBFS or relative repo path)")

# Configure Spark to use Eastern Time for all timestamps
spark.conf.set("spark.sql.session.timeZone", "America/New_York")
print("[CONFIG] Spark timezone set t4o: America/New_York (Eastern Time)")

# Resolve workspace paths
import os

# Auto-create Unity Catalog infrastructure if it doesn't exist
print("\n[SETUP] Ensuring Unity Catalog infrastructure exists...")
try:
    # Create dedicated catalog for this project
    spark.sql("CREATE CATALOG IF NOT EXISTS twitter_streaming")
    print("[SETUP] ✓ Catalog 'twitter_streaming' ready")
    
    spark.sql("CREATE SCHEMA IF NOT EXISTS twitter_streaming.default")
    print("[SETUP] ✓ Schema 'twitter_streaming.default' ready")
    
    # Create volume if it doesn't exist
    spark.sql("CREATE VOLUME IF NOT EXISTS twitter_streaming.default.checkpoints")
    print("[SETUP] ✓ Volume 'twitter_streaming.default.checkpoints' ready")
    
    print("[SETUP] ✓ All Unity Catalog infrastructure verified and ready")
except Exception as e:
    print(f"[SETUP] ⚠ Warning creating UC infrastructure: {e}")
    print("[SETUP] Note: Volume paths will be created automatically on first write")

# CRITICAL: All paths MUST be in Unity Catalog Volumes for serverless compute write access
# Using dedicated twitter_streaming catalog
incoming_dir = "/Volumes/twitter_streaming/default/checkpoints/incoming"
checkpoint_dir = "/Volumes/twitter_streaming/default/checkpoints/tweets"
checkpoint_tweets_dir = "/Volumes/twitter_streaming/default/checkpoints/tweets_raw"

# Local CSV output paths - Unity Catalog Volumes
local_csv_path = "/Volumes/twitter_streaming/default/checkpoints/aggregated_metrics.csv"
local_tweets_csv_path = "/Volumes/twitter_streaming/default/checkpoints/important_tweets.csv"

print(f"[CONFIG] Incoming directory: {incoming_dir}")
print(f"[CONFIG] Checkpoint directory: {checkpoint_dir}")
print(f"[CONFIG] Tweets checkpoint directory: {checkpoint_tweets_dir}")
print(f"[CONFIG] Local CSV path: {local_csv_path}")
print(f"[CONFIG] Local tweets CSV path: {local_tweets_csv_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2: Live Ingestion Stream Emulator (Background Thread)
# MAGIC Emulates a live streaming tweet API by breaking the Kaggle CSV into batches of 100 tweets and saving them as timestamped JSON files in the ingestion landing zone at 30-second intervals. Runs as a daemon thread with automatic shutdown after 30 minutes.

# COMMAND ----------

import threading
import time
import pandas as pd
import json
import os
import random

# Global stop event to manage background thread
stop_event = threading.Event()

def run_chunker(csv_path, output_dir, chunk_size, interval, max_duration=1800):
    """Reads a source CSV and writes chunked JSON batches to an output directory at fixed intervals.

    Args:
        csv_path: Path to the source CSV dataset.
        output_dir: Directory where JSON batch files are written.
        chunk_size: Number of tweets per batch.
        interval: Seconds between batch writes.
        max_duration: Maximum runtime in seconds before auto-shutdown.
    """
    print(f"[EMULATOR] Thread started. Reading source: {csv_path}...")
    print(f"[EMULATOR] Will run for maximum {max_duration // 60} minutes")
    emulator_start_time = time.time()
    try:
        # Robust path simplification for Python within Databricks
        real_csv_path = csv_path
        real_output_dir = output_dir
        
        # Remove any erroneous cross-prefix attempts
        if real_csv_path.startswith("dbfs:/"):
            real_csv_path = real_csv_path.replace("dbfs:/", "/dbfs/")
            
        # For Unity Catalog Volumes, use dbutils to ensure directory exists
        if real_output_dir.startswith("/Volumes/"):
            try:
                # Unity Catalog Volumes: create directory via dbutils
                dbutils.fs.mkdirs(real_output_dir)
                print(f"[EMULATOR] Created Volume directory: {real_output_dir}")
            except Exception as vol_err:
                print(f"[EMULATOR] Volume directory creation: {vol_err} (may already exist)")
        else:
            # Regular filesystem paths
            os.makedirs(real_output_dir, exist_ok=True)
        
        # Load the Twitter data
        if not os.path.exists(real_csv_path):
            print(f"[EMULATOR] Warning: Path {real_csv_path} not found. Generating dummy dataset...")
            dummy_tweets = [
                "Tesla stock is soaring to the moon! Incredible gains today. $TSLA",
                "Apple reports record-breaking quarterly earnings. $AAPL",
                "Huge market selloff incoming. Inflation rates are higher than expected.",
                "Bitcoin crashes below support level. Panic selling everywhere! $BTC"
            ]
            df = pd.DataFrame({
                "text": dummy_tweets * 50,
                "label": [random.choice([0, 1, 2]) for _ in range(200)]
            })
        else:
            df = pd.read_csv(real_csv_path)
            
        print(f"[EMULATOR] Data ready: {len(df)} rows. Commencing emulation...")
        
        chunk_idx = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not stop_event.is_set():
            # Check if max duration reached
            if time.time() - emulator_start_time >= max_duration:
                print(f"[EMULATOR] Max duration ({max_duration // 60} min) reached. Stopping.")
                break
            
            # Check if too many consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                print(f"[EMULATOR] Too many consecutive errors ({consecutive_errors}). Stopping.")
                break
            
            try:
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
                
                # Write with retry logic for transient I/O errors
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        with open(batch_file_path, "w") as f:
                            json.dump(records, f)
                        break  # Success - exit retry loop
                    except (OSError, IOError) as write_err:
                        if attempt < max_retries - 1:
                            print(f"[EMULATOR] Write error (attempt {attempt + 1}/{max_retries}): {write_err}. Retrying...")
                            time.sleep(1)  # Brief pause before retry
                        else:
                            print(f"[EMULATOR] Write failed after {max_retries} attempts: {write_err}. Skipping batch.")
                            consecutive_errors += 1
                            continue
                    
                # Success - reset error counter
                consecutive_errors = 0
                chunk_idx += 1
                time.sleep(interval)
                
            except Exception as loop_err:
                consecutive_errors += 1
                print(f"[EMULATOR] Error in batch processing loop: {loop_err}. Retrying...")
                time.sleep(2)
            
    except Exception as e:
        print(f"[EMULATOR] Error in stream emulator thread: {e}")

def start_emulator(csv_path, output_dir=None, chunk_size=50, interval=10, max_duration=1800):
    """Launches the stream emulator as a named daemon thread if not already running."""
    global stop_event
    stop_event.clear()
    
    # Resolve output directory to global incoming_dir if not specified
    output_dir = output_dir or incoming_dir
    
    # Check if thread is already running
    for t in threading.enumerate():
        if t.name == "TwitterStreamEmulator":
            print("[EMULATOR] Emulator thread already running.")
            return
            
    thread = threading.Thread(target=run_chunker, args=(csv_path, output_dir, chunk_size, interval, max_duration), name="TwitterStreamEmulator")
    thread.daemon = True
    thread.start()
    print("[EMULATOR] Ingestion emulator background thread initiated successfully.")

def stop_emulator():
    """Signals the emulator background thread to stop gracefully."""
    global stop_event
    stop_event.set()
    print("[EMULATOR] Shutdown signal transmitted. Thread will stop shortly.")

# COMMAND ----------


# Pre-run cleanup to ensure fresh workspace state
print("[INIT] Preparing workspace environment for demo run...")
print(f"[INIT] Incoming dir: {incoming_dir}")
print(f"[INIT] Checkpoint dir: {checkpoint_dir}")
print(f"[INIT] CSV output: {local_csv_path}")

try:
    # Clean up checkpoint and CSV using dbutils for Unity Catalog Volumes
    import shutil
    
    # Clear checkpoint directory (Unity Catalog Volumes)
    try:
        dbutils.fs.rm(checkpoint_dir, True)
        print("[INIT] ✓ Cleared checkpoint directory")
    except:
        print("[INIT] ✓ Checkpoint directory will be created on first use")
        
    # Clear tweets checkpoint directory (Unity Catalog Volumes)
    try:
        dbutils.fs.rm(checkpoint_tweets_dir, True)
        print("[INIT] ✓ Cleared tweets checkpoint directory")
    except:
        pass
    
    # Clear local CSV (Unity Catalog Volumes)
    try:
        dbutils.fs.rm(local_csv_path, False)
        print("[INIT] ✓ Cleared local CSV")
    except:
        print("[INIT] ✓ CSV will be created on first use")
        
    # Clear local tweets CSV (Unity Catalog Volumes)
    try:
        dbutils.fs.rm(local_tweets_csv_path, False)
        print("[INIT] ✓ Cleared local tweets CSV")
    except:
        pass
    
    # Clean incoming directory (now in Unity Catalog Volumes)
    try:
        dbutils.fs.rm(incoming_dir, True)
        print("[INIT] ✓ Cleared old incoming files")
    except:
        print("[INIT] ✓ Incoming directory will be created on first use")
    
    # Unity Catalog Volumes directories are created automatically when files are written
    print("[INIT] ✓ Unity Catalog Volumes paths configured")
    
except Exception as e:
    print(f"[INIT] ⚠ Warning during workspace initialization: {e}")

# Start the emulator (Run this to start streaming mock files)
csv_src = dbutils.widgets.get("csv_source_path")
print(f"[INIT] Starting emulator with source: {csv_src}")
start_emulator(csv_path=csv_src, output_dir=incoming_dir, chunk_size=100, interval=30, max_duration=1800)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 3: Spark Structured Streaming & NLP Preprocessing
# MAGIC Defines the JSON schema for incoming tweet batches and initializes a PySpark Structured Streaming read stream. Applies regex-based NLP normalization to remove URLs, user handles, and special characters while preserving dollar-sign (`$`) stock tickers.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType
from pyspark.sql.functions import col, regexp_replace, lower, from_unixtime

schema = StructType([
    StructField("text", StringType(), True),
    StructField("label", IntegerType(), True),
    StructField("timestamp", LongType(), True),
    StructField("followers", LongType(), True)
])

# Read streaming data from landing zone
raw_stream = (spark.readStream
              .format("json")
              .schema(schema)
              .option("maxFilesPerTrigger", 1)
              .load(incoming_dir))

# Clean tweets: remove URLs, handles, punctuation, keep currency symbols ($)
cleaned_stream = (raw_stream
    .withColumn("cleaned_text", regexp_replace(col("text"), r"http\S+|www\S+|https\S+", "")) 
    .withColumn("cleaned_text", regexp_replace(col("cleaned_text"), r"@\w+", ""))             
    .withColumn("cleaned_text", regexp_replace(col("cleaned_text"), r"[^a-zA-Z0-9\s\$]", ""))  
    .withColumn("cleaned_text", lower(col("cleaned_text")))
    .withColumn("timestamp_datetime", from_unixtime(col("timestamp")).cast("timestamp"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 4: NLTK VADER Sentiment Pandas UDF & Window Aggregation
# MAGIC Deploys a lazy-initialized VADER classifier inside a Spark Pandas UDF for parallel sentiment scoring. Computes a follower-weighted Weighted Sentiment Index (WSI) and aggregates results over 10-minute sliding windows (1-minute slide interval) with watermarking.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, window, sum, log, col, avg, count
import pandas as pd
import os

class VADERClassifier:
    """Lazy-initialized singleton wrapper for NLTK VADER SentimentIntensityAnalyzer.

    Downloads the VADER lexicon to a writable /tmp path on first access to avoid
    read-only filesystem errors on Databricks serverless workers.
    """
    _sia = None

    @classmethod
    def get_sia(cls):
        if cls._sia is None:
            import nltk
            
            # --- FIX FOR READ-ONLY FILE SYSTEM ERROR ---
            # Force NLTK to use /tmp/nltk_data which guarantees write permissions on serverless workers
            safe_download_dir = "/tmp/nltk_data"
            os.makedirs(safe_download_dir, exist_ok=True)
            if safe_download_dir not in nltk.data.path:
                nltk.data.path.append(safe_download_dir)
            
            # Download the lexicon pointing explicitly to the secure directory
            nltk.download('vader_lexicon', download_dir=safe_download_dir, quiet=True)
            # ------------------------------------------
            
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            cls._sia = SentimentIntensityAnalyzer()
        return cls._sia

@pandas_udf("positive float, negative float, neutral float, compound float")
def classify_sentiment_vader_udf(texts: pd.Series) -> pd.DataFrame:
    """Spark Pandas UDF that scores each tweet text with VADER polarity metrics."""
    sia = VADERClassifier.get_sia()
    positives, negatives, neutrals, compounds = [], [], [], []
    for text in texts:
        scores = sia.polarity_scores(str(text))
        positives.append(scores['pos'])
        negatives.append(scores['neg'])
        neutrals.append(scores['neu'])
        compounds.append(scores['compound'])
    return pd.DataFrame({
        "positive": positives,
        "negative": negatives,
        "neutral": neutrals,
        "compound": compounds
    })

# Apply UDF to streaming dataframe
sentiment_stream = (cleaned_stream
                    .withColumn("sentiment", classify_sentiment_vader_udf(col("cleaned_text")))
                    .select("*", "sentiment.*")
                    .drop("sentiment"))

# Compute follower-weighted metrics
weighted_stream = (sentiment_stream
    .withColumn("weight", log(col("followers") + 1))
    .withColumn("weighted_diff", (col("positive") - col("negative")) * col("weight"))
)

# 10-Minute Sliding Window, sliding every 1 minute
aggregated_stream = (weighted_stream
    .withWatermark("timestamp_datetime", "10 minutes")
    .groupBy(window(col("timestamp_datetime"), "10 minutes", "1 minute"))
    .agg(
        sum("weighted_diff").alias("sum_weighted_diff"),
        sum("weight").alias("sum_weight"),
        avg("negative").alias("avg_neg"),
        count("text").alias("tweet_count")
    )
    .withColumn("wsi", col("sum_weighted_diff") / col("sum_weight"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 5A: Tweet Processor Sink (`TweetProcessor`)
# MAGIC The `TweetProcessor` is a custom serializable sink designed for Spark's `foreachBatch` output. 
# MAGIC For each micro-batch, it:
# MAGIC 1. Extracts the top 10 most influential tweets (sorted by follower count).
# MAGIC 2. Merges them with the existing local history, dropping duplicate tweets (by text).
# MAGIC 3. Filters to retain tweets from the last 15 minutes, with a fallback to the top 10 overall if there are few recent tweets.
# MAGIC 4. Saves the results locally in UC Volumes and pushes them to GitHub.

# COMMAND ----------

import os

class TweetProcessor:
    """Serializable foreachBatch processor that extracts the most important tweets
    from the current micro-batch, merges them with historical important tweets,
    retains the top 10 most important tweets from the last minutes, and syncs to GitHub.
    """
    def __init__(self, csv_path, token, repo, branch, file_path):
        self.csv_path = csv_path  # Local file cache path (Unity Catalog Volume)
        self.token = token        # GitHub Personal Access Token
        self.repo = repo          # Target GitHub repository in "owner/repo" format
        self.branch = branch      # Git branch to commit and push changes to
        self.file_path = file_path # Target file path in the repository (e.g., "important_tweets.csv")

    def push_to_github(self, csv_content):
        """Pushes the updated CSV content to GitHub using the Contents API."""
        import requests
        import base64
        
        # Guard clause: bypass sync if GitHub credentials are not fully configured
        if not self.token or not self.repo or self.repo == "username/repo":
            print("[GITHUB SYNC] GitHub credentials not fully configured. Skipping tweet update.")
            return
            
        # Target URL for the GitHub Contents API
        url = f"https://api.github.com/repos/{self.repo}/contents/{self.file_path}"

        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # 1. Fetch current file SHA if it exists (necessary to overwrite files in GitHub Contents API)
        params = {"ref": self.branch} if self.branch else {}
        sha = None
        try:
            r = requests.get(url, headers=headers, params=params)
            if r.status_code == 200:
                sha = r.json().get("sha")
        except Exception as e:
            print(f"[GITHUB SYNC] Error fetching SHA from GitHub API: {e}")
            return
            
        # 2. Build upload payload: content must be base64-encoded
        content_b64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
        payload = {
            "message": "Update important financial tweets feed",
            "content": content_b64,
            "branch": self.branch
        }
        if sha:
            payload["sha"] = sha # Attach SHA to overwrite the existing file
            
        try:
            # Send PUT request to create or update the file on GitHub
            put_r = requests.put(url, headers=headers, json=payload)
            if put_r.status_code in [200, 201]:
                print(f"[GITHUB SYNC] Successfully committed and pushed updated tweets to {self.repo}/{self.file_path}")
            else:
                print(f"[GITHUB SYNC] GitHub API Error ({put_r.status_code}): {put_r.text}")
        except Exception as e:
            print(f"[GITHUB SYNC] Exception occurred during file upload: {e}")

    def __call__(self, df, batch_id):
        """Processes each micro-batch DataFrame from the streaming query."""
        if df.isEmpty():
            return
            
        print(f"\n--- Processing Tweet Micro-Batch: {batch_id} ---")
        
        from pyspark.sql.functions import col
        # Get the top 10 most followed tweets from this batch to identify high-influence messages
        top_batch = df.select("text", "followers", "timestamp", "compound").orderBy(col("followers").desc()).limit(10)
        pdf = top_batch.toPandas()
        
        # Load previously accumulated tweets to merge and maintain a sliding window
        import time
        import pandas as pd
        current_time = time.time()
        time_buffer = 15 * 60  # Retain tweets from the last 15 minutes
        
        if os.path.exists(self.csv_path):
            try:
                # Read local cached file
                old_pdf = pd.read_csv(self.csv_path)
                # Combine current batch with history and drop duplicates based on text
                combined_pdf = pd.concat([old_pdf, pdf]).drop_duplicates(subset=['text'], keep='last')
                
                # Filter to only keep tweets within our time buffer (15 minutes)
                filtered_pdf = combined_pdf[current_time - combined_pdf['timestamp'] <= time_buffer]
                
                # Fallback: if fewer than 10 tweets exist in the last 15 minutes, keep the top 10 overall
                if len(filtered_pdf) < 10:
                    pdf = combined_pdf.sort_values(by='followers', ascending=False).head(10)
                else:
                    pdf = filtered_pdf.sort_values(by='followers', ascending=False).head(10)
            except Exception as e:
                print(f"[BATCH PROCESS] Failed to merge with existing tweet history: {e}")
                pdf = pdf.sort_values(by='followers', ascending=False).head(10)
        else:
            pdf = pdf.sort_values(by='followers', ascending=False).head(10)
            
        # Save the updated list locally to the Unity Catalog Volume cache
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        pdf.to_csv(self.csv_path, index=False)
        print(f"[BATCH PROCESS] Saved important tweets locally to {self.csv_path}. Row count: {len(pdf)}")
        
        # Push updated CSV content to GitHub for the dashboard to pick up
        csv_string = pdf.to_csv(index=False)
        self.push_to_github(csv_string)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 5B: Metrics Aggregation Sink (`BatchProcessor`)
# MAGIC The `BatchProcessor` handles windowed aggregates for each micro-batch.
# MAGIC For each batch, it:
# MAGIC 1. Unpacks the window boundary timestamp correctly.
# MAGIC 2. Merges new windows with historical data, dropping duplicate windows.
# MAGIC 3. Computes the rolling Z-score of negative sentiment over the last 10 windows (sliding panic index).
# MAGIC 4. Truncates history to the last 100 windows and uploads the metrics CSV to GitHub.

# COMMAND ----------

class BatchProcessor:
    """Serializable foreachBatch processor that accumulates windowed metrics,
    computes rolling Z-scores, persists results to CSV, and syncs to GitHub.
    """
    def __init__(self, csv_path, token, repo, branch, file_path):
        self.csv_path = csv_path  # Local file cache path (Unity Catalog Volume)
        self.token = token        # GitHub Personal Access Token
        self.repo = repo          # Target GitHub repository
        self.branch = branch      # Git branch
        self.file_path = file_path # Target file path (e.g., "aggregated_metrics.csv")

    def push_to_github(self, csv_content):
        """Pushes the updated aggregated metrics CSV to GitHub Contents API."""
        import requests
        import base64
        
        # Guard clause: check credentials
        if not self.token or not self.repo or self.repo == "username/repo":
            print("[GITHUB SYNC] GitHub credentials not fully configured. Skipping repository update.")
            return
            
        url = f"https://api.github.com/repos/{self.repo}/contents/{self.file_path}"

        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # 1. Fetch current file SHA if it exists
        params = {"ref": self.branch} if self.branch else {}
        sha = None
        try:
            r = requests.get(url, headers=headers, params=params)
            if r.status_code == 200:
                sha = r.json().get("sha")
        except Exception as e:
            print(f"[GITHUB SYNC] Error fetching SHA from GitHub API: {e}")
            return
            
        # 2. Build upload payload: base64-encoded
        content_b64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
        payload = {
            "message": "Update aggregated financial sentiment seismograph metrics",
            "content": content_b64,
            "branch": self.branch
        }
        if sha:
            payload["sha"] = sha
            
        try:
            put_r = requests.put(url, headers=headers, json=payload)
            if put_r.status_code in [200, 201]:
                print(f"[GITHUB SYNC] Successfully committed and pushed updated metrics to {self.repo}/{self.file_path}")
            else:
                print(f"[GITHUB SYNC] GitHub API Error ({put_r.status_code}): {put_r.text}")
        except Exception as e:
            print(f"[GITHUB SYNC] Exception occurred during file upload: {e}")

    def __call__(self, df, batch_id):
        """Processes each micro-batch DataFrame representing windowed aggregates."""
        if df.isEmpty():
            return
            
        print(f"\n--- Processing Micro-Batch: {batch_id} ---")
        pdf = df.toPandas()
        
        # --- ROBUST WINDOW UNPACKING FIX ---
        # Spark SQL windows are returned as a Struct containing start and end timestamps.
        # This function unpacks them regardless of the format (dictionary, object, tuple).
        def get_window_bound(w, bound_type):
            if not w:
                return None
            if isinstance(w, dict):
                return w.get(bound_type)
            if hasattr(w, bound_type):
                return getattr(w, bound_type)
            # Fallback for alternative formats
            return w[0] if bound_type == 'start' else w[1]

        pdf['window_start'] = pdf['window'].apply(lambda w: get_window_bound(w, 'start'))
        pdf['window_end'] = pdf['window'].apply(lambda w: get_window_bound(w, 'end'))
        pdf = pdf.drop(columns=['window'])
        # -----------------------------------
        
        # Load previously accumulated metrics to maintain the historical trend
        if os.path.exists(self.csv_path):
            try:
                old_pdf = pd.read_csv(self.csv_path)
                # Ensure timestamps are localized identically for deduplication
                pdf['window_start'] = pd.to_datetime(pdf['window_start']).dt.tz_localize(None)
                old_pdf['window_start'] = pd.to_datetime(old_pdf['window_start']).dt.tz_localize(None)
                
                # Combine, drop duplicates keeping the most recent calculation, and sort chronologically
                combined_pdf = pd.concat([old_pdf, pdf]).drop_duplicates(subset=['window_start'], keep='last')
                pdf = combined_pdf.sort_values(by='window_start').reset_index(drop=True)
            except Exception as e:
                print(f"[BATCH PROCESS] Failed to merge with existing CSV history: {e}")
                
        # Calculate the Z-Score of negative sentiment over the last 10 windows (sliding statistics)
        if len(pdf) >= 2:
            rolling_mean = pdf['avg_neg'].rolling(window=10, min_periods=1).mean()
            rolling_std = pdf['avg_neg'].rolling(window=10, min_periods=1).std().fillna(1e-5)
            pdf['z_score'] = (pdf['avg_neg'] - rolling_mean) / rolling_std
        else:
            pdf['z_score'] = 0.0
            
        # Keep only the last 100 windows to conserve dashboard performance and limit GitHub file size
        pdf = pdf.tail(100)
        
        # Save metrics locally to Unity Catalog volume
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        pdf.to_csv(self.csv_path, index=False)
        print(f"[BATCH PROCESS] Saved metrics locally to {self.csv_path}. Row count: {len(pdf)}")
        
        # Push updated CSV content to GitHub
        csv_string = pdf.to_csv(index=False)
        self.push_to_github(csv_string)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 5C: Stream Engine Pre-Launch & Master Execution Loop
# MAGIC This cell launches the streaming queries in a sequential execution loop designed specifically for serverless compute compatibility.
# MAGIC 1. It extracts config variables from active UI widgets.
# MAGIC 2. Clears checkpoint folders to run fresh pipelines.
# MAGIC 3. Sequentially triggers `availableNow` micro-batches for both streams (`aggregated_stream` and `weighted_stream`).
# MAGIC 4. Implements graceful teardown and final cleanup of workspace directories on shutdown.

# COMMAND ----------

import time

# Step 1: Extract runtime connection tokens and configurations from active UI widgets
git_token       = dbutils.widgets.get("github_token")
git_repo        = dbutils.widgets.get("github_repo")
git_branch      = dbutils.widgets.get("github_branch")
git_file        = dbutils.widgets.get("github_file_path")
git_tweets_file = dbutils.widgets.get("github_tweets_file_path")
if not git_tweets_file or git_tweets_file.strip() == "":
    git_tweets_file = "important_tweets.csv"

print(f" -> Synchronizing target repository: {git_repo} [{git_branch}]")
print(f" -> Target metrics output file:    {git_file}")
print(f" -> Target tweets output file:     {git_tweets_file}")

# Step 2: Clear checkpoint directories for a clean processing run
# In serverless workflows, removing old checkpoints ensures we re-process from scratch
try:
    dbutils.fs.rm(checkpoint_dir, True)
    print("[CHECKPOINT] Cleared metrics checkpoint directory for fresh processing run")
except Exception as checkpoint_info:
    print(f"[CHECKPOINT] Metrics checkpoint will be created on first run: {checkpoint_info}")

try:
    dbutils.fs.rm(checkpoint_tweets_dir, True)
    print("[CHECKPOINT] Cleared tweets checkpoint directory for fresh processing run")
except Exception as checkpoint_info:
    print(f"[CHECKPOINT] Tweets checkpoint will be created on first run: {checkpoint_info}")

# Step 3: Instantiate the serializable processors
processor = BatchProcessor(
    csv_path=local_csv_path,
    token=git_token,
    repo=git_repo,
    branch=git_branch,
    file_path=git_file
)

tweet_processor = TweetProcessor(
    csv_path=local_tweets_csv_path,
    token=git_token,
    repo=git_repo,
    branch=git_branch,
    file_path=git_tweets_file
)

# Step 4: Run streaming in loop with availableNow trigger (serverless compatible)
# In Databricks Serverless, running queries continuously (ProcessingTime trigger) is very expensive.
# By triggering with 'availableNow=True', Spark processes all currently available files as a batch
# and shuts down, running in a controlled loop.
demo_duration = 1800  # Run for 30 minutes total
start_time = time.time()
processing_interval = 5  # Refresh interval in seconds

print("[DEMO] Starting streaming loop for 30-minute demo...")
print(f"[DEMO] Stream will process available data every {processing_interval} seconds and push to GitHub automatically.")

try:
    batch_count = 0
    last_progress_log = 0
    
    while time.time() - start_time < demo_duration:
        elapsed = int(time.time() - start_time)
        
        # Run one streaming micro-batch with availableNow trigger for metrics
        query = (aggregated_stream.writeStream
                 .foreachBatch(processor)
                 .outputMode("update")
                 .option("checkpointLocation", checkpoint_dir)
                 .trigger(availableNow=True)
                 .start())
        
        # Wait for this batch to complete before launching the next one
        query.awaitTermination()
        
        # Run one streaming micro-batch with availableNow trigger for raw tweets
        query_tweets = (weighted_stream.writeStream
                        .foreachBatch(tweet_processor)
                        .outputMode("append")
                        .option("checkpointLocation", checkpoint_tweets_dir)
                        .trigger(availableNow=True)
                        .start())
        
        # Wait for this batch to complete
        query_tweets.awaitTermination()
        
        batch_count += 1
        
        # Log progress every 60 seconds
        if elapsed - last_progress_log >= 60:
            minutes_elapsed = elapsed // 60
            minutes_left = (demo_duration - elapsed) // 60
            print(f"[DEMO] Elapsed: {minutes_elapsed} min | Remaining: {minutes_left} min | Batches: {batch_count}")
            last_progress_log = elapsed
        
        # Sleep until next processing interval
        time.sleep(processing_interval)
    
    print(f"\n[DEMO] Demo duration reached. Total batches processed: {batch_count}")
        
except Exception as stream_err:
    print(f"[ERROR] Streaming query execution failed: {stream_err}")
    raise stream_err

# Step 5: Graceful teardown
print("\n[DEMO] Initiating graceful shutdown...")
print(f"[DEMO] Total batches processed: {batch_count}")

print("[DEMO] Stopping Twitter stream emulator...")
stop_emulator()
time.sleep(5)

# Step 6: Final clean-up of temporary files
# Ensures the workspace remains clean, deleting temporary data and local volume paths
print("[CLEANUP] Removing generated temporary files and directory caches...")
try:
    dbutils.fs.rm(incoming_dir, True)
    dbutils.fs.rm(checkpoint_dir, True)
    dbutils.fs.rm(checkpoint_tweets_dir, True)
    if os.path.exists(local_csv_path):
        os.remove(local_csv_path)
    if os.path.exists(local_tweets_csv_path):
        os.remove(local_tweets_csv_path)
    print("[CLEANUP] Final workspace cleanup completed successfully.")
except Exception as cleanup_err:
    print(f"[WARNING] Error during final workspace cleanup: {cleanup_err}")

print("[SYSTEM] Demo run completed and workspace restored.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 6: Control & Status Utilities
# MAGIC Manual control cells for monitoring and managing the streaming pipeline. Run these individually to check stream activity, stop the streaming query, or shut down the emulator thread.

# COMMAND ----------

# Check status of the stream query
if 'query' in globals() and query.isActive:
    print(f"Streaming query is ACTIVE. Last progress: {query.lastProgress}")
else:
    print("Streaming query is INACTIVE.")

# COMMAND ----------

# Stop streaming query
if 'query' in globals() and query.isActive:
    query.stop()
    print("Streaming query stopped.")

# COMMAND ----------

# Stop emulator thread
stop_emulator()
