# Databricks notebook source
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
dbutils.widgets.text("kaggle_username", "", "Kaggle Username (Optional for dataset download)")
dbutils.widgets.text("kaggle_key", "", "Kaggle API Key (Optional for dataset download)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1.5: Kaggle Dataset Downloader
# MAGIC If you provide your Kaggle API credentials above, this cell will automatically download the real-time financial tweets dataset and extract it to DBFS at `/FileStore/tables/train_data.csv`.
# MAGIC *(To get your credentials, log into Kaggle, go to Account Settings, and click "Create New API Token" to download kaggle.json).*

# COMMAND ----------

kaggle_user = dbutils.widgets.get("kaggle_username")
kaggle_key = dbutils.widgets.get("kaggle_key")

if kaggle_user and kaggle_key:
    import os
    import json
    import subprocess
    import zipfile
    
    # Setup local folder and write credentials json
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
        json.dump({"username": kaggle_user, "key": kaggle_key}, f)
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    
    print("[KAGGLE] Downloading dataset 'sulphatet/twitter-financial-news'...")
    try:
        os.makedirs("/tmp/kaggle", exist_ok=True)
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "sulphatet/twitter-financial-news", 
            "-p", "/tmp/kaggle", 
            "--force"
        ], check=True)
        
        target_extract_dir = "/dbfs/FileStore/tables"
        os.makedirs(target_extract_dir, exist_ok=True)
        
        zip_path = "/tmp/kaggle/twitter-financial-news.zip"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_extract_dir)
            
        print(f"[KAGGLE] Success! Dataset extracted to {target_extract_dir}")
        print("Set your 'csv_source_path' widget to `/FileStore/tables/train_data.csv` to run the full dataset.")
    except Exception as e:
        print(f"[KAGGLE ERROR] Failed to download dataset: {e}")
else:
    print("[KAGGLE INFO] Kaggle credentials not provided. Skipping automated dataset download.")
    print("Using default 'csv_source_path' relative repository fallback.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2: Live Ingestion Stream Emulator (Background Thread)
# MAGIC Emulates a live streaming tweet API by breaking the Kaggle CSV into batches of 50 tweets and saving them as JSON files in `/FileStore/tweets/incoming/` at set intervals.

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
        # Convert Spark DBFS path to standard filesystem path for Pandas
        if os.path.exists(csv_path):
            real_csv_path = csv_path
        elif csv_path.startswith("dbfs:"):
            real_csv_path = csv_path.replace("dbfs:", "/dbfs")
        elif csv_path.startswith("/dbfs"):
            real_csv_path = csv_path
        elif csv_path.startswith("/Workspace"):
            real_csv_path = csv_path
        else:
            real_csv_path = "/dbfs" + csv_path if csv_path.startswith("/") else f"/dbfs/{csv_path}"
            
        real_output_dir = output_dir.replace("dbfs:", "/dbfs") if output_dir.startswith("dbfs:") else output_dir
        if not real_output_dir.startswith("/dbfs") and real_output_dir.startswith("/"):
            real_output_dir = "/dbfs" + real_output_dir
            
        os.makedirs(real_output_dir, exist_ok=True)
        
        # Load the Kaggle data
        if not os.path.exists(real_csv_path):
            # Fallback check if /dbfs mount isn't readable directly
            print(f"[EMULATOR] Warning: Path {real_csv_path} not found. Checking alternate DBFS path.")
            # We will generate mock data if the CSV doesn't exist yet, to prevent failure
            print("[EMULATOR] CSV file not found. Generating dummy dataset for execution...")
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

def start_emulator(csv_path, output_dir="/FileStore/tweets/incoming", chunk_size=50, interval=10):
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
              .load("/FileStore/tweets/incoming/"))

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

# Lazy classifier initializer for workers
class VADERClassifier:
    _sia = None

    @classmethod
    def get_sia(cls):
        if cls._sia is None:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
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

import requests
import base64
import os

# Create DBFS local storage path for metrics
dbfs_csv_path = "/dbfs/FileStore/tweets/aggregated_metrics.csv"
os.makedirs(os.path.dirname(dbfs_csv_path), exist_ok=True)

def push_to_github(csv_content, token, repo, branch, file_path):
    if not token or not repo or repo == "username/repo":
        print("[GITHUB SYNC] GitHub credentials not fully configured. Skipping repository update.")
        return
        
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # 1. Fetch current file SHA if it exists
    params = {"ref": branch} if branch else {}
    sha = None
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        sha = r.json().get("sha")
        
    # 2. Build upload payload
    content_b64 = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
    payload = {
        "message": "Update aggregated financial sentiment seismograph metrics",
        "content": content_b64,
        "branch": branch
    }
    if sha:
        payload["sha"] = sha
        
    put_r = requests.put(url, headers=headers, json=payload)
    if put_r.status_code in [200, 201]:
        print(f"[GITHUB SYNC] Successfully committed and pushed updated metrics to {repo}/{file_path}")
    else:
        print(f"[GITHUB SYNC] GitHub API Error ({put_r.status_code}): {put_r.text}")

def process_batch(df, batch_id):
    if df.isEmpty():
        return
        
    print(f"\n--- Processing Micro-Batch: {batch_id} ---")
    pdf = df.toPandas()
    
    # Unpack window struct
    pdf['window_start'] = pdf['window'].apply(lambda w: w[0] if w else None)
    pdf['window_end'] = pdf['window'].apply(lambda w: w[1] if w else None)
    pdf = pdf.drop(columns=['window'])
    
    # Load previously accumulated metrics to maintain historic trend
    if os.path.exists(dbfs_csv_path):
        try:
            old_pdf = pd.read_csv(dbfs_csv_path)
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
    pdf.to_csv(dbfs_csv_path, index=False)
    print(f"[BATCH PROCESS] Saved metrics locally. Row count: {len(pdf)}")
    
    # Push updated CSV content to GitHub
    csv_string = pdf.to_csv(index=False)
    
    # Fetch widgets credentials
    git_token = dbutils.widgets.get("github_token")
    git_repo = dbutils.widgets.get("github_repo")
    git_branch = dbutils.widgets.get("github_branch")
    git_file = dbutils.widgets.get("github_file_path")
    
    push_to_github(csv_string, git_token, git_repo, git_branch, git_file)

# COMMAND ----------

# Start Streaming Query
checkpoint_dir = "/FileStore/tweets/checkpoints"

query = (aggregated_stream.writeStream
         .foreachBatch(process_batch)
         .outputMode("update")
         .option("checkpointLocation", checkpoint_dir)
         .start())

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
