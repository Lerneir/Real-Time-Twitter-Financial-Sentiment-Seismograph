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

# =====================================================================
# NUEVA CELDA TEMPORAL DE DIAGNÓSTICO: VALIDANDO LA SOLUCIÓN COMPLETA
# =====================================================================
import os
import pandas as pd
import json
import time
import random

# 1. Parámetros directos del Workspace
test_csv_path = "/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/data/train_data.csv"
test_output_dir = "/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/incoming"

print("=== INICIANDO VERIFICACIÓN DE LA SOLUCIÓN ===")
print(f"Leyendo desde: {test_csv_path}")
print(f"Escribiendo en: {test_output_dir}")
print("-" * 50)

try:
    # 2. Forzar la creación de la carpeta directamente en el Workspace
    print("[PASO 1] Intentando crear la carpeta de destino de forma nativa...")
    os.makedirs(test_output_dir, exist_ok=True)
    print(" ✅ ¡Éxito! La carpeta se creó o ya existe sin problemas en el Workspace.")
    
    # 3. Leer el archivo CSV original
    print("\n[PASO 2] Intentando leer el archivo CSV de Twitter con pandas...")
    if os.path.exists(test_csv_path):
        df_test = pd.read_csv(test_csv_path)
        print(f" ✅ ¡Éxito! Archivo cargado. Filas encontradas: {len(df_test)}")
    else:
        print("⚠️ El archivo CSV no se encontró en esa ruta exacta. Usando datos simulados...")
        df_test = pd.DataFrame({"text": ["Sample tweet $TSLA"], "label": [1]})

    # 4. Intentar escribir un archivo JSON real de prueba
    print("\n[PASO 3] Intentando escribir un archivo JSON de prueba en la carpeta...")
    sample_records = [{
        "text": "Testing the brand new path synchronization!",
        "label": 1,
        "timestamp": int(time.time()),
        "followers": 500
    }]
    
    sample_filename = f"batch_solution_test_{int(time.time())}.json"
    sample_file_path = os.path.join(test_output_dir, sample_filename)
    
    with open(sample_file_path, "w") as f_test:
        json.dump(sample_records, f_test)
        
    print(f" ✅ ¡ÉXITO ROTUNDO! El archivo se escribió correctamente en: '{sample_file_path}'")

except Exception as e:
    print(f" ❌ FALLÓ LA SOLUCIÓN. Error capturado: {e}")

print("\n=== FIN DE LA VERIFICACIÓN ===")

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

# =====================================================================
# CELDA TEMPORAL DE DIAGNÓSTICO: PRUEBA DE ESQUEMA Y NLP DE SPARK
# =====================================================================
from pyspark.sql.functions import col, regexp_replace, lower, from_unixtime
import os

print("=== AUDITORÍA DE INGESTIÓN Y LIMPIEZA SPARK ===")
target_path = "/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/incoming/"

# 1. Verificar si hay archivos reales listos para Spark
print("[PASO 1] Comprobando disponibilidad de archivos en el Workspace...")
try:
    files = dbutils.fs.ls(target_path)
    json_files = [f for f in files if f.name.endswith('.json')]
    print(f" -> Total de archivos encontrados en la ruta: {len(files)}")
    print(f" -> Archivos JSON válidos listos para procesar: {len(json_files)}")
    if len(json_files) == 0:
        print("⚠️ ALERTA: No hay archivos JSON. Asegúrate de encender el emulador unos segundos antes.")
except Exception as path_err:
    print(f" ❌ Error al acceder a la ruta con dbutils: {path_err}")

# 2. Forzar lectura ESTÁTICA de prueba para inspección profunda
print("\n[PASO 2] Intentando leer los datos de forma estática con el esquema...")
try:
    # Usamos spark.read en lugar de readStream para poder usar .show() y recolectar logs
    test_df = spark.read.format("json").schema(schema).load(target_path)
    row_count = test_df.count()
    print(f" ✅ ¡Éxito! Spark logró leer el directorio. Registros encontrados: {row_count}")
    
    print("\n -> Estructura de tipos detectada por Spark (Schema):")
    test_df.printSchema()
    
    # 3. Validar el pipeline de limpieza (Regex)
    print("\n[PASO 3] Ejecutando transformaciones de limpieza de texto (Regex)...")
    test_cleaned_df = (test_df
        .withColumn("cleaned_text", regexp_replace(col("text"), r"http\S+|www\S+|https\S+", "")) 
        .withColumn("cleaned_text", regexp_replace(col("cleaned_text"), r"@\w+", ""))             
        .withColumn("cleaned_text", regexp_replace(col("cleaned_text"), r"[^a-zA-Z0-9\s\$]", ""))  
        .withColumn("cleaned_text", lower(col("cleaned_text")))
        .withColumn("timestamp_datetime", from_unixtime(col("timestamp")).cast("timestamp"))
    )
    
    # Mostrar una muestra de cómo quedaron los tweets limpios
    print("\n📊 MUESTRA DE DATOS PROCESADOS (Original vs Limpio):")
    if row_count > 0:
        test_cleaned_df.select("text", "cleaned_text", "followers", "timestamp_datetime").show(5, truncate=False)
    else:
        print(" No hay filas para mostrar en la muestra.")
        
except Exception as spark_err:
    print(f" ❌ EL PIPELINE DE SPARK SE ROMPIÓ. Error exacto:")
    print(str(spark_err))

print("\n=== FIN DE LA AUDITORÍA DE LA CELDA 3 ===")

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

# =====================================================================
# CELDA TEMPORAL DE DIAGNÓSTICO: VALIDACIÓN DE UDF Y VENTANA TEMPORAL
# =====================================================================
import pandas as pd
from pyspark.sql.functions import col, log, window, sum, avg, count

print("=== AUDITORÍA DE NLP VADER Y VENTANAS TEMPORALES ===")

try:
    # 1. Comprobar que dependemos del DF de prueba estático de la Celda 3
    # Si la celda de diagnóstico anterior no se ha corrido, creamos un set rápido
    if 'test_cleaned_df' not in globals():
        print("⚠️ 'test_cleaned_df' no detectado. Recreando una muestra estática rápida...")
        from pyspark.sql import Row
        import time
        sample_data = [
            Row(cleaned_text="tesla stock is soaring to the moon $tsla", followers=1000, timestamp=int(time.time())),
            Row(cleaned_text="huge market selloff panic everywhere", followers=10, timestamp=int(time.time())),
            Row(cleaned_text="apple reports record breaking earnings $aapl", followers=50000, timestamp=int(time.time()))
        ]
        test_cleaned_df = spark.createDataFrame(sample_data).withColumn("timestamp_datetime", from_unixtime(col("timestamp")).cast("timestamp"))
    
    # 2. PROBAR LA PANDAS UDF DE MANERA AISLADA
    print("\n[PASO 1] Evaluando comportamiento de la Pandas UDF en los nodos...")
    # Forzamos una ejecución local controlada de la UDF sobre nuestro DF estático
    test_sentiment_df = (test_cleaned_df
                        .withColumn("sentiment", classify_sentiment_vader_udf(col("cleaned_text")))
                        .select("*", "sentiment.*")
                        .drop("sentiment"))
    
    print(" ✅ ¡Éxito! La UDF procesó el texto y extrajo las métricas vectoriales.")
    test_sentiment_df.select("cleaned_text", "positive", "negative", "compound").show(3, truncate=False)

    # 3. VERIFICAR EL CONTEXTO MATEMÁTICO DE PESOS (Followers Weight)
    print("\n[PASO 2] Evaluando cálculo de ponderación por seguidores...")
    test_weighted_df = (test_sentiment_df
        .withColumn("weight", log(col("followers") + 1))
        .withColumn("weighted_diff", (col("positive") - col("negative")) * col("weight"))
    )
    test_weighted_df.select("followers", "weight", "weighted_diff").show(3)

    # 4. SIMULAR LA AGRUPACIÓN POR VENTANAS (Window Aggregation)
    print("\n[PASO 3] Simulando agregación por ventana de 10 minutos (Sliding Window)...")
    # Nota: Quitamos la marca de agua (.withWatermark) temporalmente aquí porque solo funciona en Streams,
    # pero ejecutamos exactamente la misma lógica de agrupación estructural.
    test_aggregated_df = (test_weighted_df
        .groupBy(window(col("timestamp_datetime"), "10 minutes", "1 minute"))
        .agg(
            sum("weighted_diff").alias("sum_weighted_diff"),
            sum("weight").alias("sum_weight"),
            avg("negative").alias("avg_neg"),
            count("cleaned_text").alias("tweet_count")
        )
        .withColumn("wsi", col("sum_weighted_diff") / col("sum_weight"))
    )
    
    print(" ✅ ¡Éxito de Estructura! Agregación calculada correctamente.")
    print("\n📊 RESULTADO MATEMÁTICO FINAL DEL SEISMOGRAPH (Estructura Estática):")
    test_aggregated_df.select("window.start", "window.end", "tweet_count", "wsi").show(5, truncate=False)

except Exception as udf_error:
    print(f" ❌ EL MÓDULO MATEMÁTICO/NLP ENCONTRÓ UN ERROR. Detalles:")
    print(str(udf_error))
    import traceback
    traceback.print_exc()

print("\n=== FIN DE LA AUDITORÍA DE LA CELDA 4 ===")

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

# =====================================================================
# TEMPORARY DIAGNOSTIC CELL: TESTING BATCH WRITER AND GITHUB REST API
# =====================================================================
import datetime
import pandas as pd
from pyspark.sql import Row

print("=== STARTING BATCH PROCESSOR & GITHUB REST API AUDIT ===")

# 1. Gather live parameter tokens from your active widgets
test_token = dbutils.widgets.get("github_token")
test_repo = dbutils.widgets.get("github_repo")
test_branch = dbutils.widgets.get("github_branch")
test_file = dbutils.widgets.get("github_file_path")

print(f"Target Repository: {test_repo}")
print(f"Target Branch:     {test_branch}")
print(f"Target File Path:   {test_file}")
print("-" * 50)

# 2. Build Mock Spark Window Aggregation Data matching Cell 4 output
print("[STEP 1] Generating mock structured window metrics...")
now = datetime.datetime.now()
mock_window = Row(start=now - datetime.timedelta(minutes=10), end=now)

# Replicating the schema produced by your streaming aggregated aggregation
mock_row = Row(
    window=mock_window,
    sum_weighted_diff=2.45,
    sum_weight=12.8,
    avg_neg=0.15,
    tweet_count=50,
    wsi=0.1914
)
mock_df = spark.createDataFrame([mock_row])

# 3. Instantiate the BatchProcessor locally
print("\n[STEP 2] Initializing BatchProcessor instance...")
test_processor = BatchProcessor(
    csv_path=dbfs_csv_path,
    token=test_token,
    repo=test_repo,
    branch=test_branch,
    file_path=test_file
)

# 4. Trigger a synchronous call to find runtime or path errors
print("\n[STEP 3] Triggering mock micro-batch execution (Synchronous test)...")
try:
    # Explicitly call __call__ directly with batch_id 999
    test_processor(mock_df, batch_id=999)
    print(" ✅ [SUCCESS] Local file generation, merging, and Z-score calculations completed flawlessly.")
    
    # Verify local file presence
    if os.path.exists(dbfs_csv_path):
        print(f" ✅ [SUCCESS] Local caching validated at: '{dbfs_csv_path}'")
        print("\n📊 Local Cache Snapshot:")
        print(pd.read_csv(dbfs_csv_path).to_string())
    else:
        print("❌ [FAILURE] Code finished but local output file was not found.")
        
except Exception as audit_err:
    print(" ❌ [FAILURE] The batch pipeline broke down during compilation or processing.")
    print(f"Error Details: {audit_err}")
    import traceback
    traceback.print_exc()

print("\n=== BATCH PROCESSOR AUDIT COMPLETE ===")

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
checkpoint_dir = "/Workspace/tmp/tweets/checkpoints"

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
        print(" ✅ [CLEANUP] Workspace scratchpad checkpoint cache wiped cleanly.")
    except Exception as cleanup_err:
        print(f" ⚠️ [CLEANUP] Notice: Skipping manual folder wipe: {cleanup_err}")

# Step 3: Instantiate the serializable micro-batch engine
processor = BatchProcessor(
    csv_path=dbfs_csv_path,
    token=git_token,
    repo=git_repo,
    branch=git_branch,
    file_path=git_file
)

# Step 4: Assemble and spark the Structured Streaming engine
print("\n🚀 Igniting Spark Structured Streaming query engine...")
try:
    query = (aggregated_stream.writeStream
             .foreachBatch(processor)
             .outputMode("update")
             .option("checkpointLocation", checkpoint_dir)
             .trigger(availableNow=True)
             .start())
    
    print(" ✅ [STREAM STARTED] Query is actively transforming live data batches.")
except Exception as stream_err:
    print(f" ❌ [CRITICAL CORRUPTION] Streaming engine execution halted. Error details: {stream_err}")

print("\n=== SYSTEM ONLINE ===")

# COMMAND ----------

# =====================================================================
# CELDA TEMPORAL DE DIAGNÓSTICO: CONTROL DE CALIDAD PRE-LANZAMIENTO
# =====================================================================
import os

print("=== AUDITORÍA FINAL DE INFRAESTRUCTURA DE STREAMING ===")
test_checkpoint_dir = "/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/checkpoints"

# 1. Auditar variables extraídas de los Widgets
print("[PASO 1] Validando credenciales cargadas desde el Driver...")
print(f" -> Token GitHub detectado (Longitud): {len(git_token) if git_token else 0} caracteres")
print(f" -> Repositorio Destino: {git_repo}")
print(f" -> Rama Activa: {git_branch}")

if not git_token or git_repo == "username/repo":
    print("⚠️ ALERTA: Las credenciales de GitHub parecen no estar configuradas en los widgets superiores.")
else:
    print(" ✅ Credenciales listas para la autenticación de la API.")

# 2. Auditar e instruir sobre el estado de los Checkpoints
print("\n[PASO 2] Analizando el directorio de Checkpoints...")
checkpoint_exists = os.path.exists(test_checkpoint_dir)
print(f" -> ¿Existe un historial de checkpoints previo en el disco?: {checkpoint_exists}")

if checkpoint_exists:
    print("\n💡 NOTA IMPORTANTE DE DEPURACIÓN:")
    print("Como corregimos la estructura interna del BatchProcessor, los checkpoints viejos")
    print("pueden causar conflictos de metadatos. Si el stream se cuelga, ejecuta la siguiente")
    print("línea en una celda para limpiar el historial y forzar un arranque en limpio:")
    print(f'   dbutils.fs.rm("{test_checkpoint_dir}", True)')
else:
    print(" ✅ Directorio limpio. Spark creará un nuevo estado inicial de metadatos.")

# 3. Validar existencia de datos en la Landing Zone
print("\n[PASO 3] Verificando combustible en la Landing Zone...")
try:
    incoming_files = dbutils.fs.ls("/Workspace/Shared/Real-Time-Twitter-Financial-Sentiment-Seismograph/tmp/tweets/incoming")
    jsons = [f for f in incoming_files if f.name.endswith('.json')]
    print(f" ✅ ¡Listo! Encontrados {len(jsons)} archivos JSON esperando a ser procesados por el stream.")
except Exception:
    print("❌ ALERTA: La carpeta 'incoming' está vacía o inaccesible. Asegúrate de encender el emulador primero.")

print("\n=== VERIFICACIÓN FINALIZADA ===")

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
