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
dbutils.widgets.text("github_repo", "username/repo", "GitHub Repository (owner/repo)")
dbutils.widgets.text("github_branch", "main", "GitHub Branch")
dbutils.widgets.text("github_file_path", "aggregated_metrics.csv", "GitHub File Path")
dbutils.widgets.text("csv_source_path", "../data/train_data.csv", "Source CSV Path (DBFS or relative repo path)")

# COMMAND ----------

# MAGIC %restart_python

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

def run_chunker(csv_path, output_dir, chunk_size, interval):
    print(f"[EMULATOR] Thread started. Reading source: {csv_path}...")
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
        while not stop_event.is_set():
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
                
            chunk_idx += 1
            time.sleep(interval)
            
    except Exception as e:
        print(f"[EMULATOR] Error in stream emulator thread: {e}")

def start_emulator(csv_path, output_dir="/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/incoming", chunk_size=50, interval=10):
    global stop_event
    stop_event.clear()
    
    # Check if thread is already running
    for t in threading.enumerate():
        if t.name == "TwitterStreamEmulator":
            print("[EMULATOR] Emulator thread already running.")
            return
            
    thread = threading.Thread(target=run_chunker, args=(csv_path, output_dir, chunk_size, interval), name="TwitterStreamEmulator")
    thread.daemon = True
    thread.start()
    print("[EMULATOR] Ingestion emulator background thread initiated successfully.")

def stop_emulator():
    global stop_event
    stop_event.set()
    print("[EMULATOR] Shutdown signal transmitted. Thread will stop shortly.")

# COMMAND ----------


# Define workspace paths
incoming_dir = "/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/incoming"
checkpoint_dir = "/Workspace/tmp/tweets/checkpoints"
local_csv_path = "/tmp/tweets/aggregated_metrics.csv"

# Pre-run cleanup to ensure fresh workspace state
print("[INIT] Preparing workspace environment for demo run...")
try:
    dbutils.fs.rm(incoming_dir, True)
    dbutils.fs.rm(checkpoint_dir, True)
    if os.path.exists(local_csv_path):
        os.remove(local_csv_path)
    print("[INIT] Pre-run cleanup completed successfully.")
except Exception as e:
    print(f"[INIT] Warning during workspace initialization: {e}")

# Start the emulator (Run this to start streaming mock files)
csv_src = dbutils.widgets.get("csv_source_path")
start_emulator(csv_path=csv_src, chunk_size=50, interval=10)

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
              .load("/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/incoming/"))

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


# MAGIC %md
# MAGIC ### Cell 5: Stream Writer & GitHub Push Synchronization

# COMMAND ----------

import os

# Create DBFS local storage path for metrics
dbfs_csv_path = "/tmp/tweets/aggregated_metrics.csv"
os.makedirs(os.path.dirname(dbfs_csv_path), exist_ok=True)

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
        
        # Save back to DBFS local cache
        pdf.to_csv(self.csv_path, index=False)
        print(f"[BATCH PROCESS] Saved metrics locally. Row count: {len(pdf)}")
        
        # Push updated CSV content to GitHub
        csv_string = pdf.to_csv(index=False)
        self.push_to_github(csv_string)

# COMMAND ----------


dbutils.fs.rm("/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/checkpoints", True)

# COMMAND ----------

# =====================================================================
# FINAL CELL: STREAM ACTIVATION AND MASTER RUN (EPHEMERAL WORKSPACE)
# =====================================================================
import os
import shutil

# SOLUTION: Use an absolute scratchpad path without the 'file:' or 'dbfs:' prefixes.
# This satisfies DatabricksCheckpointFileManager while granting streaming write access.
incoming_dir = "/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/incoming"
checkpoint_dir = "/Workspace/tmp/tweets/checkpoints"
local_csv_path = "/tmp/tweets/aggregated_metrics.csv"

print("=== INITIATING STREAM ENGINE PRE-LAUNCH CONTROL ===")

# Step 1: Extract runtime connection tokens and configurations from active UI widgets
git_token  = dbutils.widgets.get("github_token")
git_repo   = dbutils.widgets.get("github_repo")
git_branch = dbutils.widgets.get("github_branch")
git_file   = dbutils.widgets.get("github_file_path")

print(f" -> Synchronizing target repository: {git_repo} [{git_branch}]")
print(f" -> Target metrics output file:    {git_file}")

# Step 2: Clear pre-existing checkpoint metadata to ensure state consistency
if os.path.exists(checkpoint_dir):
    try:
        shutil.rmtree(checkpoint_dir)
        print("[CLEANUP] Workspace scratchpad checkpoint cache wiped cleanly.")
    except Exception as cleanup_err:
        print(f"[CLEANUP] Notice: Skipping manual folder wipe: {cleanup_err}")

# Step 3: Instantiate the serializable micro-batch engine
processor = BatchProcessor(
    csv_path=dbfs_csv_path,
    token=git_token,
    repo=git_repo,
    branch=git_branch,
    file_path=git_file
)

# Step 4: Assemble and run the Structured Streaming query in a triggered loop
import time

demo_duration = 1800  # 30 minutes
run_interval = 15     # Interval in seconds between stream triggers
start_time = time.time()
elapsed = 0
last_progress_log = 0

print("[DEMO] Starting 30-minute triggered execution loop...")

try:
    while elapsed < demo_duration:
        # Start the query with trigger(availableNow=True) to process all available files
        query = (aggregated_stream.writeStream
                 .foreachBatch(processor)
                 .outputMode("update")
                 .option("checkpointLocation", checkpoint_dir)
                 .trigger(availableNow=True)
                 .start())
        
        # Await completion of this triggered run
        query.awaitTermination()
        
        # Update elapsed time
        current_time = time.time()
        elapsed = int(current_time - start_time)
        
        # Log progress every 60 seconds
        if elapsed - last_progress_log >= 60:
            minutes_left = (demo_duration - elapsed) // 60
            print(f"[DEMO] Elapsed: {elapsed // 60} min | Remaining: {minutes_left} min")
            last_progress_log = elapsed
            
        # Pause before triggering the next run
        time.sleep(run_interval)
        
except Exception as stream_err:
    print(f"[ERROR] Triggered stream execution failed: {stream_err}")
    raise stream_err

# Step 6: Graceful teardown
print("[DEMO] Demo duration reached or stream stopped. Initiating shutdown...")

if 'query' in globals() and query.isActive:
    print("[DEMO] Stopping Spark streaming query...")
    query.stop()
    query.awaitTermination(timeout=30)
    print("[DEMO] Spark streaming query stopped successfully.")

print("[DEMO] Stopping Twitter stream emulator...")
stop_emulator()
time.sleep(5)

# Step 7: Final clean-up of temporary files
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
