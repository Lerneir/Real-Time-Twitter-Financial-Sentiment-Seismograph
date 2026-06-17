import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import requests
import datetime

# --- Premium Page Configurations ---
st.set_page_config(
    page_title="Financial Sentiment Seismograph",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection for sleek dark mode & cards
st.markdown("""
<style>
    /* Styling main background and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek card container */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 255, 127, 0.3);
    }
    
    /* Custom signals */
    .signal-buy {
        border-left: 6px solid #00ff7f;
        background: linear-gradient(90deg, rgba(0, 255, 127, 0.1) 0%, rgba(0, 0, 0, 0) 100%);
    }
    .signal-sell {
        border-left: 6px solid #ff4b4b;
        background: linear-gradient(90deg, rgba(255, 75, 75, 0.1) 0%, rgba(0, 0, 0, 0) 100%);
    }
    .signal-hold {
        border-left: 6px solid #ffaa00;
        background: linear-gradient(90deg, rgba(255, 170, 0, 0.1) 0%, rgba(0, 0, 0, 0) 100%);
    }
    
    .signal-title {
        font-size: 1.1rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    
    .signal-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 5px;
        transition: font-size 0.3s ease;
    }

    .signal-description {
        margin-top: 10px;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #dddddd;
    }

    .kpi-card {
        height: auto;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 24px;
        transition: min-height 0.3s ease;
    }

    .kpi-title {
        color: #888888;
        font-size: 1.1rem;
        letter-spacing: 1px;
        font-weight: 600;
        margin-top: 5px;
    }

    .kpi-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 15px 0;
        transition: font-size 0.3s ease;
    }

    .kpi-desc {
        color: #aaaaaa;
        font-size: 0.9rem;
        line-height: 1.4;
    }

    /* Tweet Card & Feed Containers */
    .tweet-feed-container {
        max-height: 550px;
        overflow-y: auto;
        padding-right: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        background: rgba(0, 0, 0, 0.1);
        padding: 12px;
    }

    .tweet-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        transition: border-color 0.3s ease;
    }
    .tweet-card:hover {
        border-color: rgba(29, 161, 242, 0.2);
    }

    .tweet-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
        flex-wrap: wrap;
        gap: 8px;
    }

    .tweet-user-info {
        display: flex;
        align-items: center;
    }

    .tweet-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1DA1F2 0%, #0d8ecf 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 1rem;
        margin-right: 12px;
        box-shadow: 0 4px 10px rgba(29, 161, 242, 0.2);
        flex-shrink: 0;
    }

    .tweet-meta {
        display: flex;
        flex-direction: column;
    }

    .tweet-username-row {
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .tweet-username {
        color: #ffffff;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .tweet-handle {
        color: #8899A6;
        font-size: 0.8rem;
    }

    .tweet-right-side {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .tweet-time {
        color: #8899A6;
        font-size: 0.8rem;
    }

    .tweet-body {
        color: #e1e8ed;
        font-size: 0.95rem;
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .panic-log-card {
        background: rgba(255, 75, 75, 0.08);
        border: 1px solid rgba(255, 75, 75, 0.2);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
    }
    
    .pulse {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #00ff7f;
        box-shadow: 0 0 0 0 rgba(0, 255, 127, 0.7);
        animation: pulse 1.5s infinite;
        vertical-align: middle;
        margin-right: 8px;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 255, 127, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(0, 255, 127, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 255, 127, 0);
        }
    }

    /* --- Responsive Media Queries --- */
    @media (max-width: 992px) {
        .kpi-card {
            min-height: auto;
            height: auto;
            gap: 12px;
        }
        .kpi-value {
            font-size: 2.8rem;
            margin: 10px 0;
        }
        .kpi-title {
            font-size: 0.95rem;
        }
    }

    @media (max-width: 768px) {
        .metric-card {
            padding: 18px;
        }
        .signal-value {
            font-size: 2rem;
        }
        .signal-description {
            font-size: 1rem;
        }
        .kpi-value {
            font-size: 2.5rem;
        }
    }

    @media (max-width: 600px) {
        .tweet-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 6px;
        }
        .tweet-right-side {
            width: 100%;
            justify-content: space-between;
            margin-top: 4px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 8px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Title Header ---
st.title("⚡ Real-Time Twitter Financial Sentiment Seismograph")
st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Control Hub")

# Section 1: Panic Threshold settings (Top priority)
st.sidebar.subheader("🚨 Sentiment Alerts")
z_threshold = st.sidebar.number_input(
    "Panic Threshold (Z-Score σ)",
    min_value=0.0,
    max_value=10.0,
    value=2.5,
    step=0.1,
    help="Define the standard deviation threshold of negative sentiment to trigger panic/sell signals."
)

# Section 2: Ingestion Pipeline
st.sidebar.subheader("📡 Ingestion Pipeline")
data_source = st.sidebar.selectbox(
    "Active Data Stream Source",
    ["Mock Stream (Local Emulation)", "Live Databricks Pipeline (via GitHub)", "Offline File Inspection (CSV)"],
    index=1
)

# Configuration variables
github_url = ""
uploaded_file = None
github_token = ""

if data_source == "Live Databricks Pipeline (via GitHub)":
    st.sidebar.markdown("### GitHub Repository Sync")
    repo = st.sidebar.text_input("Repository Path (owner/repo)", "Lerneir/Real-Time-Twitter-Financial-Sentiment-Seismograph")
    github_token = st.sidebar.text_input("GitHub Token (optional)", type="password", help="Providing a token enables instant updates (bypassing GitHub's 5-minute CDN cache) and increases API rate limits.")
    file_path = st.sidebar.text_input("Metrics File Name", "aggregated_metrics.csv")
    tweets_file_name = st.sidebar.text_input("Tweets File Name", "important_tweets.csv")
    branch = st.sidebar.text_input("Target Branch", "main")
    
    if repo and file_path:
        github_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"
        st.sidebar.markdown(f'<div style="font-size: 0.8rem; color: #8899A6; margin-bottom: 6px;">📈 Metrics stream: <span style="color: #1DA1F2; word-break: break-all;">{github_url}</span></div>', unsafe_allow_html=True)
    if repo and tweets_file_name:
        github_tweets_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{tweets_file_name}"
        st.sidebar.markdown(f'<div style="font-size: 0.8rem; color: #8899A6;">🐦 Tweets stream: <span style="color: #1DA1F2; word-break: break-all;">{github_tweets_url}</span></div>', unsafe_allow_html=True)
elif data_source == "Offline File Inspection (CSV)":
    uploaded_file = st.sidebar.file_uploader("Upload 'aggregated_metrics.csv'", type=["csv"])

# Section 3: Display & Refresh Settings
st.sidebar.subheader("🎚️ Display & Refresh Settings")
auto_refresh = st.sidebar.toggle("Enable Live Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Frequency (seconds)", min_value=2, max_value=30, value=5)

# Section 4: Session Maintenance
if st.sidebar.button("🗑️ Reset Session & Clear Data"):
    st.session_state.clear()
    st.success("Session data reset successfully!")
    time.sleep(1)
    st.rerun()

# --- Initialize Demo Data in Session State ---
if 'demo_df' not in st.session_state:
    # Generate initial history
    base_time = datetime.datetime.now() - datetime.timedelta(minutes=60)
    history = []
    wsi_val = 0.05
    avg_neg_val = 0.12
    
    for i in range(40):
        w_start = base_time + datetime.timedelta(minutes=i)
        w_end = w_start + datetime.timedelta(minutes=10)
        
        # Random walk for sentiment index
        wsi_val = np.clip(wsi_val + np.random.normal(0, 0.08), -0.8, 0.8)
        avg_neg_val = np.clip(avg_neg_val + np.random.normal(0.001, 0.02), 0.02, 0.45)
        
        history.append({
            "window_start": w_start,
            "window_end": w_end,
            "sum_weighted_diff": wsi_val * 1000,
            "sum_weight": 1000.0,
            "avg_neg": avg_neg_val,
            "tweet_count": np.random.randint(150, 600),
            "wsi": wsi_val,
        })
        
    df = pd.DataFrame(history)
    
    # Calculate Z-score
    rolling_mean = df['avg_neg'].rolling(window=10, min_periods=1).mean()
    rolling_std = df['avg_neg'].rolling(window=10, min_periods=1).std().fillna(1e-5)
    df['z_score'] = (df['avg_neg'] - rolling_mean) / rolling_std
    
    st.session_state['demo_df'] = df

if 'demo_tweets' not in st.session_state:
    # Generate initial 10 mock important tweets
    mock_texts = [
        "Major institutional buyers spotted loading up on $TSLA. Volume is spiking! 🚀",
        "Analysts raise target price for $AAPL following incredible customer demand indicators.",
        "Fed suggests rate cuts might be delayed. Bond yields ticking up. $SPY",
        "Bitcoin holding strong above key support levels. Bullish structure remains intact. $BTC",
        "Rumors of merger between key semiconductor players driving tech sector rally. $SOXX",
        "Oil prices surge amid geopolitical tensions, raising inflation concerns. $USO",
        "Retail sales report beats expectations, signaling resilient consumer spending.",
        "Prominent venture capitalist predicts major breakthrough in consumer AI models soon.",
        "Market volatility index VIX climbs as traders hedge against upcoming CPI release.",
        "Unbelievable earnings beat from top SaaS player. Guidance exceeds top-end analyst estimates."
    ]
    
    initial_tweets = []
    base_time = datetime.datetime.now()
    for i, text in enumerate(mock_texts):
        followers = int(np.random.randint(5000, 450000))
        if any(w in text for w in ["spike", "rally", "beat", "strong"]):
            compound = np.random.uniform(0.4, 0.85)
        elif any(w in text for w in ["delay", "tension", "volatility"]):
            compound = np.random.uniform(-0.7, -0.3)
        else:
            compound = np.random.uniform(-0.1, 0.3)
            
        initial_tweets.append({
            "timestamp": int((base_time - datetime.timedelta(minutes=i*2)).timestamp()),
            "text": text,
            "followers": followers,
            "compound": compound
        })
    st.session_state['demo_tweets'] = pd.DataFrame(initial_tweets)

def fetch_from_github_api(repo, branch, file_path, token=None):
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    if token and token.strip() != "":
        headers["Authorization"] = f"token {token.strip()}"
        
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    params = {"ref": branch} if branch else {}
    
    try:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code == 200:
            import base64
            content_b64 = r.json().get("content", "")
            content_b64 = content_b64.replace("\n", "").replace("\r", "")
            content_bytes = base64.b64decode(content_b64)
            return content_bytes.decode("utf-8"), None
        else:
            return None, f"HTTP {r.status_code}"
    except Exception as e:
        return None, str(e)

def fetch_raw_github(github_url):
    try:
        cache_bypassed_url = f"{github_url}?t={int(time.time())}"
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        r = requests.get(cache_bypassed_url, headers=headers)
        if r.status_code == 200:
            return r.text, None
        return None, f"HTTP {r.status_code}"
    except Exception as e:
        return None, str(e)

# --- Fetch and Clean Metrics Data ---
def load_data():
    if data_source == "Mock Stream (Local Emulation)":
        # Check if we need to append a new row (emulating streaming update)
        df = st.session_state['demo_df']
        tweets_df = st.session_state['demo_tweets']
        
        # Add new point if last point is older than interval
        last_row = df.iloc[-1]
        time_diff = (datetime.datetime.now() - pd.to_datetime(last_row['window_start'])).total_seconds()
        
        # Simulating fresh stream ingestion
        if time_diff > refresh_interval:
            new_start = pd.to_datetime(last_row['window_start']) + datetime.timedelta(minutes=1)
            new_end = new_start + datetime.timedelta(minutes=10)
            
            # Occasionally simulate a panic spike
            is_spike = np.random.rand() > 0.90
            
            if is_spike:
                wsi_val = np.random.uniform(-0.6, -0.4)
                avg_neg_val = np.random.uniform(0.35, 0.55) # Clear spike
            else:
                wsi_val = np.clip(last_row['wsi'] + np.random.normal(0, 0.06), -0.7, 0.7)
                avg_neg_val = np.clip(last_row['avg_neg'] + np.random.normal(0, 0.02), 0.02, 0.35)
                
            new_row = {
                "window_start": new_start,
                "window_end": new_end,
                "sum_weighted_diff": wsi_val * 1000,
                "sum_weight": 1000.0,
                "avg_neg": avg_neg_val,
                "tweet_count": np.random.randint(150, 600),
                "wsi": wsi_val,
            }
            
            df = pd.concat([df, pd.DataFrame([new_row])]).reset_index(drop=True)
            
            # Recalculate Z-score
            rolling_mean = df['avg_neg'].rolling(window=10, min_periods=1).mean()
            rolling_std = df['avg_neg'].rolling(window=10, min_periods=1).std().fillna(1e-5)
            df['z_score'] = (df['avg_neg'] - rolling_mean) / rolling_std
            
            # Cap history
            df = df.tail(100)
            st.session_state['demo_df'] = df
            
            # Simulating fresh tweet update
            new_tweet_texts = [
                "Whale transaction alert: 50,000 BTC moved off exchanges. Bullish signal? $BTC",
                "CPI data comes in cooler than expected. Stock futures rallying! $DIA $QQQ",
                "Tech stock dump: Insiders selling shares at record pace. Be careful.",
                "Earnings alert: $NVDA beats on EPS and revenue, stock jumps 8% after hours!",
                "BREAKING: Regulatory investigations launched against major tech giants.",
                "Retail interest in meme stocks reaches 12-month high. Volatility spiking.",
                "Bond yields fall to 3-month low as investors seek safe havens.",
                "Short seller releases critical report on top growth stock. Shares down 15%."
            ]
            selected_text = np.random.choice(new_tweet_texts)
            
            if is_spike or any(w in selected_text.lower() for w in ["dump", "investigation", "down", "crisis"]):
                new_compound = np.random.uniform(-0.8, -0.3)
            elif any(w in selected_text.lower() for w in ["rally", "beat", "jump", "btc"]):
                new_compound = np.random.uniform(0.3, 0.8)
            else:
                new_compound = np.random.uniform(-0.2, 0.2)
                
            new_tweet = {
                "timestamp": int(datetime.datetime.now().timestamp()),
                "text": selected_text,
                "followers": int(np.random.randint(10000, 800000)),
                "compound": new_compound
            }
            
            tweets_df = pd.concat([pd.DataFrame([new_tweet]), tweets_df]).drop_duplicates(subset=['text']).head(10)
            st.session_state['demo_tweets'] = tweets_df
            
        return df, tweets_df, None
        
    elif data_source == "Live Databricks Pipeline (via GitHub)":
        if not repo or repo.strip() == "" or repo == "username/repo":
            return None, None, "Please configure your actual GitHub repository credentials in the sidebar."
        try:
            # Fetch metrics CSV
            if github_token and github_token.strip() != "":
                csv_text, err = fetch_from_github_api(repo, branch, file_path, github_token)
                if err:
                    return None, None, f"Failed to fetch metrics via GitHub API: {err}. Check your token and repo path."
            else:
                github_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"
                csv_text, err = fetch_raw_github(github_url)
                if err:
                    return None, None, f"Failed to fetch metrics via Raw URL: {err}. Consider entering a GitHub Token in the sidebar to bypass caching and prevent rate limits."
            
            import io
            df = pd.read_csv(io.StringIO(csv_text))
            # Basic validation
            required_cols = ['window_start', 'window_end', 'wsi', 'avg_neg', 'z_score', 'tweet_count']
            if not all(col in df.columns for col in required_cols):
                return None, None, f"CSV is missing required metrics columns. Expected: {required_cols}"
            
            df['window_start'] = pd.to_datetime(df['window_start'])
            df['window_end'] = pd.to_datetime(df['window_end'])
            
            # Fetch tweets from GitHub
            tweets_df = None
            if repo and tweets_file_name:
                if github_token and github_token.strip() != "":
                    tweets_csv_text, err_tweets = fetch_from_github_api(repo, branch, tweets_file_name, github_token)
                    if not err_tweets:
                        try:
                            tweets_df = pd.read_csv(io.StringIO(tweets_csv_text))
                        except Exception as e:
                            print(f"Error parsing tweets: {e}")
                else:
                    tweets_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{tweets_file_name}"
                    tweets_csv_text, err_tweets = fetch_raw_github(tweets_url)
                    if not err_tweets:
                        try:
                            tweets_df = pd.read_csv(io.StringIO(tweets_csv_text))
                        except Exception as e:
                            print(f"Error parsing tweets: {e}")
            
            # Validate tweets_df columns if loaded
            if tweets_df is not None:
                required_tweets_cols = ['text', 'followers', 'timestamp', 'compound']
                if not all(col in tweets_df.columns for col in required_tweets_cols):
                    tweets_df = None
            
            return df.sort_values(by='window_start').reset_index(drop=True), tweets_df, None
        except Exception as e:
            return None, None, f"Connection/Parsing Error: {str(e)}"
            
    elif data_source == "Offline File Inspection (CSV)":
        if uploaded_file is None:
            return None, None, "Please drag & drop or select an aggregated_metrics.csv file in the sidebar."
        try:
            df = pd.read_csv(uploaded_file)
            df['window_start'] = pd.to_datetime(df['window_start'])
            df['window_end'] = pd.to_datetime(df['window_end'])
            
            # Fallback to demo tweets for local file mode
            tweets_df = st.session_state.get('demo_tweets')
            return df.sort_values(by='window_start').reset_index(drop=True), tweets_df, None
        except Exception as e:
            return None, None, f"File parsing error: {str(e)}"

def render_tweet_card(text, followers, timestamp, compound):
    # Determine sentiment badge
    if compound >= 0.05:
        sentiment_html = '<span style="background-color: rgba(0, 255, 127, 0.15); color: #00ff7f; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: 10px; border: 1px solid rgba(0, 255, 127, 0.2);">🟢 Bullish</span>'
    elif compound <= -0.05:
        sentiment_html = '<span style="background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: 10px; border: 1px solid rgba(255, 75, 75, 0.2);">🔴 Bearish</span>'
    else:
        sentiment_html = '<span style="background-color: rgba(255, 170, 0, 0.15); color: #ffaa00; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: 10px; border: 1px solid rgba(255, 170, 0, 0.2);">🟡 Neutral</span>'
    
    # Format follower count
    if followers >= 1000000:
        follower_str = f"{followers / 1000000:.1f}M"
    elif followers >= 1000:
        follower_str = f"{followers / 1000:.1f}K"
    else:
        follower_str = str(followers)
        
    # Format timestamp
    if isinstance(timestamp, str):
        try:
            dt = pd.to_datetime(timestamp)
            time_str = dt.strftime("%I:%M %p")
        except:
            time_str = timestamp
    elif hasattr(timestamp, "strftime"):
        time_str = timestamp.strftime("%I:%M %p")
    else:
        # timestamp is unix epoch
        try:
            if timestamp > 1e11:
                dt = datetime.datetime.fromtimestamp(timestamp / 1000)
            else:
                dt = datetime.datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%I:%M %p")
        except:
            time_str = str(timestamp)
            
    # Mock handle and name based on follower count to make it look realistic
    handles = ["MarketMover", "FinTwitAlpha", "CryptoSentinel", "MacroPulse", "VolumeAlerts", "WSB_Intelligence", "GrowthTrends"]
    handle_idx = int(followers) % len(handles)
    username = handles[handle_idx]
    
    tweet_html = f"""
    <div class="tweet-card">
        <div class="tweet-header">
            <div class="tweet-user-info">
                <div class="tweet-avatar">
                    {username[0]}
                </div>
                <div class="tweet-meta">
                    <div class="tweet-username-row">
                        <span class="tweet-username">{username}</span>
                        <svg viewBox="0 0 24 24" aria-label="Verified account" style="width: 14px; height: 14px; fill: #1DA1F2;"><g><path d="M22.5 12.5c0-1.58-.875-2.95-2.148-3.6.154-.435.238-.905.238-1.4 0-2.21-1.71-3.99-3.818-3.99-.48 0-.94.1-1.348.27C14.825 2.515 13.512 1.5 12 1.5s-2.825 1.015-3.422 2.28c-.408-.17-.868-.27-1.348-.27-2.108 0-3.818 1.78-3.818 3.99 0 .495.084.965.238 1.4-1.273.65-2.148 2.02-2.148 3.6 0 1.58.875 2.95 2.148 3.6-.154.435-.238.905-.238 1.4 0 2.21 1.71 3.99 3.818 3.99.48 0 .94-.1 1.348-.27.597 1.265 1.91 2.28 3.422 2.28s2.825-1.015 3.422-2.28c.408.17.868.27 1.348.27 2.108 0 3.818-1.78 3.818-3.99 0-.495-.084-.965-.238-1.4 1.273-.65 2.148-2.02 2.148-3.6zm-12.72 3.19l-3.26-3.26 1.41-1.42 1.84 1.83 4.97-4.97 1.42 1.42-6.38 6.4z"></path></g></svg>
                    </div>
                    <span class="tweet-handle">@{username.lower()} · {follower_str} followers</span>
                </div>
            </div>
            <div class="tweet-right-side">
                <span class="tweet-time">{time_str}</span>
                {sentiment_html}
            </div>
        </div>
        <div class="tweet-body">{text}</div>
    </div>
    """
    return tweet_html

# Load the dataframes
df, tweets_df, error_msg = load_data()

if error_msg:
    st.error(error_msg)
    st.info("💡 Tip: You can switch to 'Demo Mode' in the sidebar to visualize a self-generating live stream immediately!")
else:
    # Display running banner
    st.markdown(f"<div><span class='pulse'></span><b>Pipeline Status:</b> Active & Listening (Refreshing every {refresh_interval}s)</div>", unsafe_allow_html=True)
    st.write("")
    
    # Retrieve latest metrics
    latest_row = df.iloc[-1]
    latest_wsi = float(latest_row['wsi'])
    latest_z = float(latest_row['z_score'])
    latest_neg = float(latest_row['avg_neg'])
    latest_count = int(latest_row['tweet_count'])
    
    # Determine trading signal & thresholds
    # Panic signal: Z-score > 2.5
    # Buy signal: WSI > 0.25 (Positive Market Sentiment)
    # Sell/Panic: Z-score > 2.5 or WSI < -0.25 (Bearish/High Volatility Alert)
    if latest_z > z_threshold:
        signal = "SELL (PANIC DETECTED)"
        signal_class = "signal-sell"
        signal_color = "#ff4b4b"
        advisor_text = f"🚨 MARKET ALERT: Sentiment Z-Score exceeds panic threshold (+{z_threshold:.2f}). Social media negativity is spiking abnormally. High risk of liquidity/selloff event. Reallocate assets to defensive positions."
    elif latest_wsi > 0.25:
        signal = "BUY (BULLISH)"
        signal_class = "signal-buy"
        signal_color = "#00ff7f"
        advisor_text = "📈 BULLISH SYNC: Sentiment Index is positive (>0.25) with stable panic levels. Retail buyers are active. Favorable momentum detected for tech & speculative assets."
    else:
        signal = "HOLD (NEUTRAL)"
        signal_class = "signal-hold"
        signal_color = "#ffaa00"
        advisor_text = "⚖️ RANGE BOUND: Weighted Sentiment Index remains range-bound (-0.25 to 0.25) and Z-score is normal. Market is consolidating. Maintain current portfolio holdings."
        
    # --- Top Row: Signal Advisor Banner ---
    st.markdown(f"""
    <div class="metric-card {signal_class}" style="text-align: left; margin-bottom: 25px; padding: 24px;">
        <div class="signal-title">Automated Trading Signal</div>
        <div class="signal-value" style="color: {signal_color};">{signal}</div>
        <p class="signal-description">
            {advisor_text}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Second Row: KPI Cards & Dial Gauge ---
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # WSI Metric Card
        st.markdown(f"""
        <div class="metric-card kpi-card">
            <div class="kpi-title">WEIGHTED SENTIMENT INDEX (WSI)</div>
            <div class="kpi-value" style="color: {'#00ff7f' if latest_wsi > 0 else '#ff4b4b'};">
                {latest_wsi:+.4f}
            </div>
            <div class="kpi-desc">
                Follower-Weighted Difference between Positive and Negative tweet lexicons.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Z-Score Metric Card
        st.markdown(f"""
        <div class="metric-card kpi-card">
            <div class="kpi-title">NEGATIVITY Z-SCORE</div>
            <div class="kpi-value" style="color: {'#ff4b4b' if latest_z > 2.5 else '#aaaaaa'};">
                {latest_z:+.2f}
            </div>
            <div class="kpi-desc">
                Number of Standard Deviations the current negative sentiment is above its 10-window rolling mean.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Gauge Seismometer Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_wsi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Fear & Greed Dial (WSI)", 'font': {'size': 16, 'color': '#888888'}},
            gauge = {
                'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#888888"},
                'bar': {'color': signal_color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#444444",
                'steps': [
                    {'range': [-1, -0.25], 'color': 'rgba(255, 75, 75, 0.15)'},
                    {'range': [-0.25, 0.25], 'color': 'rgba(255, 170, 0, 0.05)'},
                    {'range': [0.25, 1], 'color': 'rgba(0, 255, 127, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#ff4b4b", 'width': 4},
                    'thickness': 0.75,
                    'value': -0.25
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#ffffff", 'family': "Outfit"},
            margin=dict(l=20, r=20, t=40, b=20),
            height=280
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    st.write("")
    
    # --- Third Row: Historical Sentiment Seismograph Chart ---
    st.subheader("🌋 Historical Sentiment Seismograph")
    
    # Dual-axis plotly chart to represent "seismograph" activity
    fig = go.Figure()
    
    # Add WSI Line
    fig.add_trace(go.Scatter(
        x=df['window_start'],
        y=df['wsi'],
        name="Sentiment Index (WSI)",
        mode='lines+markers',
        line=dict(color="#1DA1F2", width=2.5),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor='rgba(29, 161, 242, 0.03)'
    ))
    
    # Add Z-Score Line
    fig.add_trace(go.Scatter(
        x=df['window_start'],
        y=df['z_score'],
        name="Negativity Z-Score",
        mode='lines',
        line=dict(color="#ff4b4b", width=1.8, dash='dash'),
    ))
    
    # Add Panic Threshold Line
    fig.add_trace(go.Scatter(
        x=df['window_start'],
        y=[z_threshold]*len(df),
        name=f"Panic Threshold (Z={z_threshold:.2f})",
        mode='lines',
        line=dict(color="rgba(255, 75, 75, 0.4)", width=1.5, dash='dot'),
        showlegend=True
    ))

    fig.update_layout(
        paper_bgcolor='rgba(255,255,255,0.02)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            title="Window Start Time"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            title="Value / Metric Scale"
        ),
        margin=dict(l=40, r=40, t=30, b=40),
        height=450,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Fourth Row: Twitter Radar & Historical Logs ---
    col_bottom_left, col_bottom_right = st.columns([11, 9])
    
    with col_bottom_left:
        st.subheader("🐦 Real-Time Tweet Radar (High Influence)")
        if tweets_df is None or len(tweets_df) == 0:
            st.info("ℹ️ No high influence tweets data available. Start the Databricks pipeline to generate and push `important_tweets.csv` to GitHub.")
        else:
            # Render scrollable tweet container
            tweets_html_list = []
            for _, row in tweets_df.iterrows():
                tweets_html_list.append(render_tweet_card(
                    text=row['text'],
                    followers=int(row['followers']),
                    timestamp=row['timestamp'],
                    compound=float(row['compound'])
                ))
            
            feed_html = f"""
            <div class="tweet-feed-container">
                {"".join(tweets_html_list)}
            </div>
            """
            st.markdown(feed_html, unsafe_allow_html=True)
            
    with col_bottom_right:
        st.subheader("📋 Live Activity Logs")
        
        # Display the table of the last 5 window outputs (reduced from 10 for compact look)
        table_df = df[['window_start', 'window_end', 'wsi', 'avg_neg', 'z_score', 'tweet_count']].copy()
        table_df = table_df.sort_values(by='window_start', ascending=False).head(5)
        table_df.columns = ["Start Time", "End Time", "WSI (Sentiment)", "Avg Negative", "Z-Score", "Volume (Tweets)"]
        
        st.dataframe(
            table_df.style.format({
                "WSI (Sentiment)": "{:+.4f}",
                "Avg Negative": "{:.4f}",
                "Z-Score": "{:+.2f}",
                "Volume (Tweets)": "{:,}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.write("")
        st.subheader("🔔 Panic Trigger History")
        
        panics = df[df['z_score'] > z_threshold].copy()
        if len(panics) == 0:
            st.info("✅ No market panic events recorded in the current window history.")
        else:
            panics = panics.sort_values(by='window_start', ascending=False)
            panic_html_list = []
            for _, row in panics.head(3).iterrows():
                time_str = pd.to_datetime(row['window_start']).strftime("%Y-%m-%d %H:%M:%S")
                panic_html_list.append(f"""
                <div class="panic-log-card">
                    <span style="color: #ff4b4b; font-weight: bold;">⚠️ {time_str}</span><br>
                    <span style="font-size: 0.9rem; color: #dddddd;">Z-Score: <b>{row['z_score']:.2f}</b> | Negativity: <b>{row['avg_neg']:.4f}</b></span>
                </div>
                """)
            st.markdown("".join(panic_html_list), unsafe_allow_html=True)

# Auto-refresh loop
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
