# Real-Time Twitter Financial Sentiment Seismograph

An end-to-end cloud-based streaming analytics pipeline and real-time dashboard designed for institutional portfolio managers and quantitative traders. This platform emulates a live social media feed, processes and cleans data at scale, classifies sentiment using NLP lexicons, computes statistical market panic flags, and serves active trading signals via a responsive frontend.

This project was built for **DAMO630 – Advanced Data Analytics** at the Master of Data Analytics program, University of Niagara Falls Canada.

------------------------------------------------------------------------

## 📖 Course Modules Integrated

1.  **Mining Data Streams (Week 3-4):** Implements a PySpark Structured Streaming read stream monitoring an active ingestion folder (`/tmp/tweets/incoming/`). A background thread chunker divides the Twitter financial dataset into micro-batches to emulate a live, high-frequency JSON API feed.
2.  **Natural Language Processing (NLP) (Week 5):** Uses Spark SQL regex transformation functions to normalize raw tweet text (scrubbing URLs, user handles, trailing emojis, and punctuation) in parallel prior to sentiment classification.
3.  **Sentiment Analysis (Week 6):** Deploys NLTK's **VADER (Valence Aware Dictionary and sEntiment Reasoner)** inside a Spark Pandas UDF (User Defined Function). Polarity scores (positive, negative, neutral, and compound) are inferred on worker nodes.
4.  **Statistical Signal Analytics (Advanced Application):** Computes a log-weighted **Weighted Sentiment Index (WSI)** incorporating follower counts. A rolling Z-score tracks negative sentiment variance over a sliding 10-window range to trigger automated **BUY**, **SELL**, or **HOLD** decisions when negativity crosses a critical $+2.5\sigma$ panic threshold.

------------------------------------------------------------------------

## 🛠️ System Architecture

```         
                                  [ Databricks Community Edition ]
                                +----------------------------------+
                                | 1. Ingestion Stream Emulator     |
                                |    (Reads Kaggle CSV -> JSONs)   |
                                +-----------------+----------------+
                                                  |
                                                  v (file:/tmp/tweets/incoming)
                                +-----------------+----------------+
                                | 2. PySpark Structured Stream    |
                                |    (Reads JSONs as live stream)  |
                                +-----------------+----------------+
                                                  |
                                                  v
                                +-----------------+----------------+
                                | 3. NLP Normalization Pipeline    |
                                |    (Clean URLs, handles, symbols)|
                                +-----------------+----------------+
                                                  |
                                                  v
                                +-----------------+----------------+
                                | 4. NLTK VADER Pandas UDF         |
                                |    (Parallel Worker Inference)   |
                                +-----------------+----------------+
                                                  |
                                                  v
                                +-----------------+----------------+
                                | 5. Sliding Windows & Z-Score     |
                                |    (10m Window, 1m Slide)        |
                                +-----------------+----------------+
                                                  |
                                                  v (ForeachBatch Sync)
                                +-----------------+----------------+
                                | 6. GitHub REST API Committer     |
                                |    (Updates aggregated CSV)      |
                                +-----------------+----------------+
                                                  |
                                                  v (Public raw.githubusercontent.com URL)
                                     [ Streamlit Community Cloud ]
                                +-----------------+----------------+
                                | 7. Sleek Dark-Mode Dashboard     |
                                |    (Plots Seismograph + Signals) |
                                +----------------------------------+
```

------------------------------------------------------------------------

## 🚀 Deployment & Setup Instructions

### Part 1: Setting up Databricks Community Edition

1.  **Upload Dataset:**
    -   Download the `twitter-financial-news` dataset from Kaggle (specifically the `train_data.csv` containing columns `text` and `label`).
    -   Log into [Databricks Community Edition](https://community.cloud.databricks.com/).
    -   Go to **Catalog** -\> **Create Table** -\> Upload your `train_data.csv` to DBFS.
    -   Note the uploaded DBFS file path (typically `/FileStore/tables/train_data.csv`).
2.  **Import Notebook:**
    -   Create a new Python Notebook in Databricks.
    -   Copy the contents of [`databricks_notebook.py`](notebooks/databricks_notebook.py) and import or paste it into your workspace.
3.  **Configure Workspace Credentials:**
    -   In your Databricks Notebook header, you will see input fields (Widgets) generated automatically:
        -   `github_token`: Paste your GitHub Personal Access Token (PAT) with `repo` read/write scopes.
        -   `github_repo`: Enter your repository name in `owner/repository` format.
        -   `github_branch`: Enter target branch (typically `main`).
        -   `github_file_path`: Target file path for metrics (typically `aggregated_metrics.csv`).
        -   `csv_source_path`: The DBFS location of your uploaded Kaggle CSV.
4.  **Execute Pipeline:**
    -   Click **Run All** in the Databricks notebook.
    -   The first cell installs dependencies (`nltk`, `requests`, `gitpython`).
    -   Cell 2 boots the background emulator thread, which periodically creates tweet chunks.
    -   Cells 3 & 4 initiate the PySpark Stream.
    -   Cell 5 uses `foreachBatch` to append metrics, compute rolling Z-scores, and sync directly with your GitHub repository.

------------------------------------------------------------------------

### Part 2: Running the Dashboard and Tests Locally

1.  **Install Dependencies:** Ensure you have Python 3.9+ installed. Run the following command in your terminal:

    ``` bash
    pip install -r requirements.txt
    ```

2.  **Launch Streamlit:** Start the web application server locally:

    ``` bash
    streamlit run src/dashboard/app.py
    ```

    A browser window will open automatically at `http://localhost:8501`. 
    
    *Note: The dashboard can also be easily deployed to [Streamlit Community Cloud](https://streamlit.io/cloud) by linking this repository directly.*

3.  **Run Pipeline Tests:** You can verify the modular sentiment classification, emulation, and NLP components by executing the test suite:

    ``` bash
    python -m tests.test_pipeline
    ```

4.  **Explore Dashboard Modes:**

    -   **Demo Mode (Self-Generating Stream):** Default out-of-the-box mode. Does not require active Databricks connections. The dashboard simulates active stream feeds and feeds synthetic sentiment walks to verify signals, gauges, and historical plots instantly.
    -   **Databricks Pipeline:** Fetches metrics updated in real-time by your Databricks cluster using your public GitHub RAW CSV URL.
    -   **Local File Upload:** Drag and drop an `aggregated_metrics.csv` to review custom offline datasets.