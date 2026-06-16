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

# COMMAND ----------

# MAGIC %pip install nltk

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.fs.rm("/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/checkpoints", True)

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
dbutils.widgets.text("csv_source_path", "/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/data/train_data.csv", "Source CSV Path (DBFS or relative repo path)")

# Configure Spark to use Eastern Time for all timestamps
spark.conf.set("spark.sql.session.timeZone", "America/New_York")
print("[CONFIG] Spark timezone set to: America/New_York (Eastern Time)")

# Resolve workspace paths
import os

# Use absolute paths directly - incoming files can be in /Workspace/ (read-only is OK for reading)
incoming_dir = "/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/incoming"

# CRITICAL: Checkpoint directory MUST be in a writable location for Spark state stores
# Using Unity Catalog Volumes - writable on serverless compute
checkpoint_dir = "/Volumes/twitter_streaming/default/checkpoints/tweets"

# Local CSV output path - Unity Catalog Volumes
local_csv_path = "/Volumes/twitter_streaming/default/checkpoints/aggregated_metrics.csv"

print(f"[CONFIG] Incoming directory: {incoming_dir}")
print(f"[CONFIG] Checkpoint directory: {checkpoint_dir}")
print(f"[CONFIG] Local CSV path: {local_csv_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2: Live Ingestion Stream Emulator (Background Thread)
# MAGIC Emulates a live streaming tweet API by breaking the Kaggle CSV into batches of 50 tweets and saving them as JSON files in `/tmp/tweets/incoming/` at set intervals.

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
    print(f"[EMULATOR] Thread started. Reading source: {csv_path}...")
    print(f"[EMULATOR] Will run for maximum {max_duration // 60} minutes")
    emulator_start_time = time.time()
    try:
        # Robust path simplification for Python within Databricks Workspace
        real_csv_path = csv_path
        real_output_dir = output_dir
        
        # Remove any erroneous cross-prefix attempts
        if real_csv_path.startswith("dbfs:/"):
            real_csv_path = real_csv_path.replace("dbfs:/", "/dbfs/")
            
        # Ensure the destination directory exists in the Workspace without touching /dbfs
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
    
    # Clear local CSV (Unity Catalog Volumes)
    try:
        dbutils.fs.rm(local_csv_path, False)
        print("[INIT] ✓ Cleared local CSV")
    except:
        print("[INIT] ✓ CSV will be created on first use")
    
    # Clean and create incoming directory (this one is in /Workspace/)
    import shutil
    if os.path.exists(incoming_dir):
        shutil.rmtree(incoming_dir)
        print("[INIT] ✓ Cleared old incoming files")
    
    # Create full directory path including all parent directories
    os.makedirs(incoming_dir, exist_ok=True)
    
    # Verify the directory was actually created
    if not os.path.exists(incoming_dir):
        raise Exception(f"Failed to create incoming directory: {incoming_dir}")
    
    # Verify write permissions by creating a test file
    test_file = os.path.join(incoming_dir, ".test_write")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"[INIT] ✓ Incoming directory initialized with write permissions: {incoming_dir}")
    except Exception as perm_err:
        raise Exception(f"No write permissions in incoming directory: {incoming_dir}. Error: {perm_err}")
    
    # Unity Catalog Volumes directories are created automatically - no need to create them
    print("[INIT] ✓ Unity Catalog Volumes paths configured")
    
    # Verify incoming directory is ready
    if os.path.exists(incoming_dir):
        existing_files = os.listdir(incoming_dir)
        print(f"[INIT] ✓ Incoming directory has {len(existing_files)} existing files")
    
except Exception as e:
    print(f"[INIT] ⚠ Warning during workspace initialization: {e}")

# Start the emulator (Run this to start streaming mock files)
csv_src = dbutils.widgets.get("csv_source_path")
print(f"[INIT] Starting emulator with source: {csv_src}")
start_emulator(csv_path=csv_src, output_dir=incoming_dir, chunk_size=100, interval=30, max_duration=1800)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 3: Spark Structured Streaming & NLP Preprocessing

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

# MAGIC
# MAGIC %md
# MAGIC ### Cell 4: NLTK VADER Sentiment Pandas UDF & Window Aggregation

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, window, sum, log, col, avg, count
import pandas as pd
import os

# Lazy classifier initializer for workers with write-safe paths
class VADERClassifier:
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

# MAGIC
# MAGIC %md
# MAGIC ### Cell 5: Stream Writer & GitHub Push Synchronization

# COMMAND ----------

import os

# Use local_csv_path directly - directory creation happens in BatchProcessor
dbfs_csv_path = local_csv_path

class BatchProcessor:
    def __init__(self, csv_path, token, repo, branch, file_path):
        self.csv_path = csv_path
        self.token = token
        self.repo = repo
        self.branch = branch
        self.file_path = file_path

    def push_to_github(self, csv_content):
        import requests
        import base64
        
        if not self.token or not self.repo or self.repo == "username/repo":
            print("[GITHUB SYNC] GitHub credentials not fully configured. Skipping repository update.")
            return
            
        url = f"https://api.github.com/repos/{self.repo}/contents/{self.file_path}"
        print(f"{url}")
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
            
        # 2. Build upload payload
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
        if df.isEmpty():
            return
            
        print(f"\n--- Processing Micro-Batch: {batch_id} ---")
        pdf = df.toPandas()
        
        # --- ROBUST WINDOW UNPACKING FIX ---
        # Handles both dictionary format {"start": ..., "end": ...} and traditional tuple/list formats
        def get_window_bound(w, bound_type):
            if not w:
                return None
            if isinstance(w, dict):
                return w.get(bound_type)
            if hasattr(w, bound_type):
                return getattr(w, bound_type)
            # Fallback for old-school tuple/list formats
            return w[0] if bound_type == 'start' else w[1]

        pdf['window_start'] = pdf['window'].apply(lambda w: get_window_bound(w, 'start'))
        pdf['window_end'] = pdf['window'].apply(lambda w: get_window_bound(w, 'end'))
        pdf = pdf.drop(columns=['window'])
        # -----------------------------------
        
        # Load previously accumulated metrics to maintain historic trend
        import os
        if os.path.exists(self.csv_path):
            try:
                import pandas as pd
                old_pdf = pd.read_csv(self.csv_path)
                # Ensure timestamp columns are parsed identically
                pdf['window_start'] = pd.to_datetime(pdf['window_start']).dt.tz_localize(None)
                old_pdf['window_start'] = pd.to_datetime(old_pdf['window_start']).dt.tz_localize(None)
                
                # Combine, drop duplicates keeping the most recent calculation, and sort
                combined_pdf = pd.concat([old_pdf, pdf]).drop_duplicates(subset=['window_start'], keep='last')
                pdf = combined_pdf.sort_values(by='window_start').reset_index(drop=True)
            except Exception as e:
                print(f"[BATCH PROCESS] Failed to merge with existing CSV history: {e}")
                
        # Calculate Z-Score of negative sentiment over the last 10 windows
        if len(pdf) >= 2:
            rolling_mean = pdf['avg_neg'].rolling(window=10, min_periods=1).mean()
            rolling_std = pdf['avg_neg'].rolling(window=10, min_periods=1).std().fillna(1e-5)
            pdf['z_score'] = (pdf['avg_neg'] - rolling_mean) / rolling_std
        else:
            pdf['z_score'] = 0.0
            
        # Keep only the last 100 windows to conserve dashboard performance and stay within API constraints
        pdf = pdf.tail(100)
        
        # Save back to local cache - ensure directory exists
        import os
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        pdf.to_csv(self.csv_path, index=False)
        print(f"[BATCH PROCESS] Saved metrics locally to {self.csv_path}. Row count: {len(pdf)}")
        
        # Push updated CSV content to GitHub
        csv_string = pdf.to_csv(index=False)
        self.push_to_github(csv_string)

# COMMAND ----------

# DBTITLE 1,Cell 18
# =====================================================================
# FINAL CELL: STREAM ACTIVATION AND MASTER RUN (SERVERLESS COMPATIBLE)
# =====================================================================
import os
import time

print("=== INITIATING STREAM ENGINE PRE-LAUNCH CONTROL ===")

# Step 1: Extract runtime connection tokens and configurations from active UI widgets
git_token  = dbutils.widgets.get("github_token")
git_repo   = dbutils.widgets.get("github_repo")
git_branch = dbutils.widgets.get("github_branch")
git_file   = dbutils.widgets.get("github_file_path")

print(f" -> Synchronizing target repository: {git_repo} [{git_branch}]")
print(f" -> Target metrics output file:    {git_file}")

# Step 2: Preserve checkpoint to enable incremental processing
# NOTE: Checkpoint allows availableNow to process only NEW files, not all files every time
try:
    # Only clear checkpoint if explicitly needed for a fresh start
    # Uncomment the next line ONLY if you want to reprocess all data from scratch:
    dbutils.fs.rm(checkpoint_dir, True)
    print("[CHECKPOINT] Using existing checkpoint for incremental processing (avoids reprocessing old files)")
except Exception as checkpoint_info:
    print(f"[CHECKPOINT] Checkpoint will be created on first run: {checkpoint_info}")

# Step 3: Instantiate the serializable micro-batch engine
processor = BatchProcessor(
    csv_path=dbfs_csv_path,
    token=git_token,
    repo=git_repo,
    branch=git_branch,
    file_path=git_file
)

# Step 4: Run streaming in loop with availableNow trigger (serverless compatible)
demo_duration = 1800  # 30 minutes
start_time = time.time()
processing_interval = 5  # Process every 5 seconds

print("[DEMO] Starting streaming loop for 30-minute demo...")
print(f"[DEMO] Stream will process available data every {processing_interval} seconds and push to GitHub automatically.")

try:
    batch_count = 0
    last_progress_log = 0
    
    while time.time() - start_time < demo_duration:
        elapsed = int(time.time() - start_time)
        
        # Run one streaming micro-batch with availableNow trigger
        query = (aggregated_stream.writeStream
                 .foreachBatch(processor)
                 .outputMode("update")
                 .option("checkpointLocation", checkpoint_dir)
                 .trigger(availableNow=True)
                 .start())
        
        # Wait for this batch to complete
        query.awaitTermination()
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
print("[CLEANUP] Removing generated temporary files and directory caches...")
try:
    dbutils.fs.rm(incoming_dir, True)
    dbutils.fs.rm(checkpoint_dir, True)
    if os.path.exists(local_csv_path):
        os.remove(local_csv_path)
    print("[CLEANUP] Final workspace cleanup completed successfully.")
except Exception as cleanup_err:
    print(f"[WARNING] Error during final workspace cleanup: {cleanup_err}")

print("[SYSTEM] Demo run completed and workspace restored.")

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ### Cell 6: Control and Status Utilities
# MAGIC Run these cells to check stream activity, view logs, or shut down queries and threads cleanly.

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
