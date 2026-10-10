import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from pathlib import Path

st.set_page_config(page_title="EV Charging Monitor", layout="wide")

BASE = Path(__file__).parent
model = joblib.load(BASE / 'rf_model.pkl')

# ---------------- Styling ----------------
st.markdown("""
<style>
.banner {
    background: linear-gradient(90deg, #0f766e 0%, #1d4ed8 100%);
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 18px;
}
.banner h1 { color: white; margin: 0; font-size: 1.9rem; }
.banner p  { color: #e0f2fe; margin: 4px 0 0 0; font-size: 0.95rem; }
.section-label {
    font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase;
    opacity: 0.65; margin: 18px 0 8px 2px; font-weight: 600;
}
.cp-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 2px; }
.cp-sub   { font-size: 0.8rem; opacity: 0.65; margin-bottom: 10px; }
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    padding: 10px 12px;
    border-radius: 10px;
}
button[data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------------- Data ----------------
@st.cache_data
def load_data():
    csv_path = BASE / '202410DatasetEVOfficeParking_v0.csv'
    if not csv_path.exists():
        st.error("Dataset not found. Download it from "
                 "https://data.4tu.nl/datasets/80ef3824-3f5d-4e45-8794-3b8791efbd13 "
                 "and place the CSV next to app.py.")
        st.stop()
    df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig')    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]

    required = ['EV_id_x', 'start_datetime', 'end_datetime', 'total_energy', 'evse_uid']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in the dataset file: {missing}")
        st.write("Columns found in the file:", df.columns.tolist())
        st.stop()

    df['start_datetime'] = pd.to_datetime(df['start_datetime'], dayfirst=True)
    df['end_datetime'] = pd.to_datetime(df['end_datetime'], dayfirst=True)
    df['duration_min'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds() / 60
    df = df[df['duration_min'] > 0].copy()
    df['avg_power_kw'] = df['total_energy'] / (df['duration_min'] / 60)
    return df


@st.cache_data
def build_stream(n_normal=25, n_attack=8, seed=7):
    rng = np.random.default_rng(seed)
    df = load_data()

    normal = df.sample(n_normal, random_state=seed).copy()
    normal['truth'] = 'normal'

    attack = df.sample(n_attack, random_state=seed + 1).copy()
    kinds = rng.choice(['FDI', 'DDoS', 'MITM'], size=n_attack)
    for idx, kind in zip(attack.index, kinds):
        if kind == 'FDI':
            attack.loc[idx, 'total_energy'] *= rng.uniform(1.5, 3)
        elif kind == 'DDoS':
            attack.loc[idx, 'duration_min'] *= rng.uniform(0.3, 0.5)
            attack.loc[idx, 'avg_power_kw'] = attack.loc[idx, 'total_energy'] / (attack.loc[idx, 'duration_min'] / 60)
        else:
            attack.loc[idx, 'avg_power_kw'] *= rng.uniform(-0.5, 0.5)
    attack['truth'] = ['attack (' + k + ')' for k in kinds]

    return pd.concat([normal, attack]).sample(frac=1, random_state=seed).reset_index(drop=True)


df = load_data()

# ---------------- Header ----------------
st.markdown("""
<div class="banner">
  <h1>EV Charging Infrastructure Monitor</h1>
  <p>Cyberattack detection for EV charging sessions using machine learning</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Stations Overview", "Live Monitoring"])

# ================= TAB 1: Overview =================
with tab1:
    st.markdown('<div class="section-label">Dataset summary</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container(border=True):
            st.metric("Total sessions recorded", f"{len(df):,}")
    with s2:
        with st.container(border=True):
            st.metric("Charge points in dataset", f"{df['evse_uid'].nunique():,}")
    with s3:
        with st.container(border=True):
            st.metric("Data period",
                      f"{df['start_datetime'].min():%b %Y} - {df['start_datetime'].max():%b %Y}")

    st.markdown('<div class="section-label">Busiest charge points</div>', unsafe_allow_html=True)
    st.caption("Recorded historical charging activity of the 4 busiest charge points.")

    top4 = df['evse_uid'].value_counts().head(4).index.tolist()
    cols = st.columns(4)

    for col, uid in zip(cols, top4):
        d = df[df['evse_uid'] == uid]
        with col:
            with st.container(border=True):
                st.markdown(f'<div class="cp-title">Charge Point {uid}</div>'
                            f'<div class="cp-sub">Historical activity</div>',
                            unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                m1.metric("Sessions", f"{len(d):,}")
                m2.metric("Energy (kWh)", f"{d['total_energy'].sum():,.0f}")
                st.metric("Avg session length (min)", f"{d['duration_min'].mean():.0f}")

                st.divider()
                st.caption("Monthly energy delivered (kWh)")
                monthly = d.set_index('start_datetime')['total_energy'].resample('MS').sum()
                st.line_chart(monthly, height=170)

                st.divider()
                st.caption("Latest sessions")
                latest = (d.sort_values('start_datetime', ascending=False)
                            .head(6)[['EV_id_x', 'start_datetime', 'total_energy', 'duration_min']]
                            .rename(columns={'EV_id_x': 'EV', 'start_datetime': 'Start',
                                             'total_energy': 'kWh', 'duration_min': 'Min'}))
                latest['kWh'] = latest['kWh'].round(1)
                latest['Min'] = latest['Min'].round(0)
                st.dataframe(latest, hide_index=True, use_container_width=True)

# ================= TAB 2: Live monitoring =================
with tab2:
    st.markdown('<div class="section-label">Control panel</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.write("Simulated real-time stream: sessions arrive one by one and each is checked "
                 "automatically by the trained Random Forest model.")
        c1, c2 = st.columns([3, 1])
        with c1:
            speed = st.slider("Seconds between sessions", 0.2, 3.0, 1.0)
        with c2:
            st.write("")
            start = st.button("Start Monitoring", use_container_width=True, type="primary")

    stream = build_stream()
    live = st.empty()

    if start:
        log = []
        flagged = 0

        for i, row in stream.iterrows():
            X = pd.DataFrame([[row['total_energy'], row['duration_min'], row['avg_power_kw']]],
                             columns=['total_energy', 'duration_min', 'avg_power_kw'])
            pred = int(model.predict(X)[0])
            flagged += pred

            log.append({
                'EV': row['EV_id_x'],
                'Station': row['evse_uid'],
                'Energy (kWh)': round(row['total_energy'], 2),
                'Duration (min)': round(row['duration_min'], 1),
                'Power (kW)': round(row['avg_power_kw'], 2),
                'Model says': 'ANOMALY' if pred == 1 else 'Normal',
                'Actual (simulation)': row['truth'],
            })

            with live.container():
                st.markdown('<div class="section-label">Live status</div>', unsafe_allow_html=True)
                k1, k2, k3 = st.columns(3)
                with k1:
                    with st.container(border=True):
                        st.metric("Sessions monitored", f"{i + 1} / {len(stream)}")
                with k2:
                    with st.container(border=True):
                        st.metric("Normal", i + 1 - flagged)
                with k3:
                    with st.container(border=True):
                        st.metric("Anomalies flagged", flagged)

                st.markdown('<div class="section-label">Current session</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"**EV {row['EV_id_x']}** at charge point **{row['evse_uid']}**")
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Energy", f"{row['total_energy']:.2f} kWh")
                    d2.metric("Duration", f"{row['duration_min']:.1f} min")
                    d3.metric("Avg power", f"{row['avg_power_kw']:.2f} kW")
                    if pred == 1:
                        st.error("ANOMALY DETECTED - suspicious charging behavior flagged for review.")
                    else:
                        st.success("Session normal - no issues detected.")

                st.markdown('<div class="section-label">Session log</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.dataframe(pd.DataFrame(log).iloc[::-1], hide_index=True,
                                 use_container_width=True)

            time.sleep(speed)

        st.info("Monitoring complete. The log shows the model's decision next to the ground truth "
                "(known only because the attacks are simulated).")
