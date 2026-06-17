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
</style>
""", unsafe_allow_html=True)

# --- Title Header ---
st.title("⚡ Real-Time Twitter Financial Sentiment Seismograph")
st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.image("https://img.icons8.com/nolan/128/seismometer.png", width=70)
st.sidebar.header("🕹️ Control & Data Panel")

data_source = st.sidebar.selectbox(
    "Choose Data Pipeline Mode",
    ["Demo Mode (Self-Generating Stream)", "Databricks Pipeline (GitHub CSV)", "Local File Upload"],
    index=1
)

# Configuration for GitHub connection
github_url = ""
uploaded_file = None

if data_source == "Databricks Pipeline (GitHub CSV)":
    st.sidebar.markdown("### GitHub Sync Configuration")
    repo = st.sidebar.text_input("GitHub Repo (owner/repo)", "Lerneir/Real-Time-Twitter-Financial-Sentiment-Seismograph")
    file_path = st.sidebar.text_input("CSV File Name", "aggregated_metrics.csv")
    branch = st.sidebar.text_input("Branch", "main")
    
    if repo and file_path:
        github_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"
        st.sidebar.caption(f"Fetching from: `{github_url}`")
elif data_source == "Local File Upload":
    uploaded_file = st.sidebar.file_uploader("Upload 'aggregated_metrics.csv'", type=["csv"])

# Auto-refresh toggles
auto_refresh = st.sidebar.toggle("Auto-Refresh Dashboard", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=2, max_value=30, value=5)

z_threshold = st.sidebar.number_input(
    "Negativity Z-Score Panic Threshold (σ)",
    min_value=0.0,
    max_value=10.0,
    value=2.5,
    step=0.1,
    help="Manually enter or adjust the standard deviation threshold for triggering sell/panic signals."
)

# Clear demo data button
if st.sidebar.button("Reset Session / Demo Data"):
    st.session_state.clear()
    st.success("Session state cleared!")
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

# --- Fetch and Clean Metrics Data ---
def load_data():
    if data_source == "Demo Mode (Self-Generating Stream)":
        # Check if we need to append a new row (emulating streaming update)
        df = st.session_state['demo_df']
        
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
            
        return df, None
        
    elif data_source == "Databricks Pipeline (GitHub CSV)":
        if not github_url or not repo or repo.strip() == "" or repo == "username/repo":
            return None, "Please configure your actual GitHub repository credentials in the sidebar."
        try:
            # Prevent caching by appending a timestamp and using cache-busting headers
            cache_bypassed_url = f"{github_url}?t={int(time.time())}"
            headers = {
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
            r = requests.get(cache_bypassed_url, headers=headers)
            if r.status_code != 200:
                return None, f"Failed to fetch CSV from GitHub: HTTP {r.status_code}. Ensure the file exists and repository is public."
            
            import io
            df = pd.read_csv(io.StringIO(r.text))
            # Basic validation
            required_cols = ['window_start', 'window_end', 'wsi', 'avg_neg', 'z_score', 'tweet_count']
            if not all(col in df.columns for col in required_cols):
                return None, f"CSV is missing required metrics columns. Expected: {required_cols}"
            
            df['window_start'] = pd.to_datetime(df['window_start'])
            df['window_end'] = pd.to_datetime(df['window_end'])
            return df.sort_values(by='window_start').reset_index(drop=True), None
        except Exception as e:
            return None, f"Connection/Parsing Error: {str(e)}"
            
    elif data_source == "Local File Upload":
        if uploaded_file is None:
            return None, "Please drag & drop or select an aggregated_metrics.csv file in the sidebar."
        try:
            df = pd.read_csv(uploaded_file)
            df['window_start'] = pd.to_datetime(df['window_start'])
            df['window_end'] = pd.to_datetime(df['window_end'])
            return df.sort_values(by='window_start').reset_index(drop=True), None
        except Exception as e:
            return None, f"File parsing error: {str(e)}"

# Load the dataframe
df, error_msg = load_data()

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
    <div class="metric-card {signal_class}" style="text-align: left; margin-bottom: 25px;">
        <div class="signal-title">Automated Trading Signal</div>
        <div class="signal-value" style="color: {signal_color};">{signal}</div>
        <p style="margin-top: 10px; font-size: 1.1rem; line-height: 1.6; color: #dddddd;">
            {advisor_text}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Second Row: KPI Cards & Dial Gauge ---
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # WSI Metric Card
        st.markdown(f"""
        <div class="metric-card" style="height: 280px;">
            <h3 style="color: #888888; font-size: 1.1rem; letter-spacing: 1px; margin-top: 10px;">WEIGHTED SENTIMENT INDEX (WSI)</h3>
            <h1 style="font-size: 3.5rem; font-weight: 800; margin: 15px 0; color: {'#00ff7f' if latest_wsi > 0 else '#ff4b4b'};">
                {latest_wsi:+.4f}
            </h1>
            <p style="color: #aaaaaa; font-size: 0.9rem;">
                Follower-Weighted Difference between Positive and Negative tweet lexicons.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Z-Score Metric Card
        st.markdown(f"""
        <div class="metric-card" style="height: 280px;">
            <h3 style="color: #888888; font-size: 1.1rem; letter-spacing: 1px; margin-top: 10px;">NEGATIVITY Z-SCORE</h3>
            <h1 style="font-size: 3.5rem; font-weight: 800; margin: 15px 0; color: {'#ff4b4b' if latest_z > 2.5 else '#aaaaaa'};">
                {latest_z:+.2f}
            </h1>
            <p style="color: #aaaaaa; font-size: 0.9rem;">
                Number of Standard Deviations the current negative sentiment is above its 10-window rolling mean.
            </p>
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
        line=dict(color="#00ffff", width=2.5),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 255, 0.03)'
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

    # --- Fourth Row: Historical Log & Alert Table ---
    col_log1, col_log2 = st.columns([2, 1])
    
    with col_log1:
        st.subheader("📋 Live Activity Logs")
        
        # Display the table of the last 10 window outputs
        table_df = df[['window_start', 'window_end', 'wsi', 'avg_neg', 'z_score', 'tweet_count']].copy()
        # Sort desc for viewer convenience
        table_df = table_df.sort_values(by='window_start', ascending=False).head(10)
        table_df.columns = ["Start Time", "End Time", "WSI (Sentiment)", "Avg Negative", "Z-Score", "Volume (Tweets)"]
        
        # Format decimal values
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
        
    with col_log2:
        st.subheader("🔔 Panic Trigger History")
        
        panics = df[df['z_score'] > z_threshold].copy()
        if len(panics) == 0:
            st.info("✅ No market panic events recorded in the current window history.")
        else:
            panics = panics.sort_values(by='window_start', ascending=False)
            for _, row in panics.head(5).iterrows():
                time_str = pd.to_datetime(row['window_start']).strftime("%Y-%m-%d %H:%M:%S")
                st.warning(f"⚠️ **{time_str}** | Z-Score: **{row['z_score']:.2f}** (Negativity: {row['avg_neg']:.4f})")

# Auto-refresh loop
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
