import os
import sys
import yaml
import torch
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Custom module imports
from preprocessing.csv_loader import load_flow_csv
from preprocessing.pcap_parser import parse_pcap_file
from preprocessing.window_builder import build_network_states, create_sequences
from preprocessing.graph_builder import build_window_graph
from preprocessing.scaler import StateScaler
from models.lstm_world_model import TemporalLSTMWorldModel
from models.temporal_gnn_world_model import TemporalGNNWorldModel
from models.graph_encoder import GraphEncoder
from models.baseline_model import LogisticRegressionBaseline
from forecasting.rollout import perform_k_step_rollout
from forecasting.gnn_rollout import perform_gnn_k_step_rollout
from forecasting.risk_engine import calculate_network_risk
from forecasting.stage_mapping import get_stage_details, STAGE_DESCRIPTIONS
from explainability.shap_explainer import ModelExplainer
from explainability.graph_explainer import explain_network_graph
from monitoring.drift_detector import DistributionDriftDetector

st.set_page_config(
    page_title="CyberForecaster SOC | Network Attack Forecasting",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark cybersecurity SOC UI theme
st.markdown("""
<style>
    /* Dark SOC Theme Variables & Base */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Cards & Containers */
    .soc-card {
        background: #151C28;
        border: 1px solid #232D3F;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    
    .kpi-card {
        background: #151C28;
        border-radius: 8px;
        padding: 16px 18px;
        border-left: 4px solid #38BDF8;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    .threat-normal { border-left: 5px solid #22C55E !important; }
    .threat-elevated { border-left: 5px solid #EAB308 !important; }
    .threat-warning { border-left: 5px solid #F97316 !important; }
    .threat-critical { border-left: 5px solid #EF4444 !important; }

    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #F8FAFC;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #64748B;
        margin-top: 4px;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0F141C !important;
        border-right: 1px solid #1E293B;
    }
    
    .sidebar-section-header {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #38BDF8;
        text-transform: uppercase;
        margin-top: 16px;
        margin-bottom: 8px;
    }

    /* Custom Badges */
    .status-badge-online {
        background-color: #052E16;
        color: #4ADE80;
        border: 1px solid #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .file-status-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 12px;
        color: #CBD5E1;
        margin-top: 8px;
    }

    /* Custom Horizontal Rules */
    hr {
        border-color: #1E293B !important;
        margin: 16px 0 !important;
    }

    /* Table & Plot Overrides */
    div[data-testid="stTable"] table {
        background-color: #151C28;
        color: #E2E8F0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_all_models(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Baseline LSTM World Model
    lstm_model = TemporalLSTMWorldModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        num_stages=config['model']['num_stages']
    ).to(device)

    weights_path = config['model']['weights_path']
    if os.path.exists(weights_path):
        lstm_model.load_state_dict(torch.load(weights_path, map_location=device))
        lstm_model.eval()

    # 2. Advanced Temporal GNN World Model
    gnn_model = TemporalGNNWorldModel(
        node_dim=10,
        graph_embed_dim=64,
        state_dim=23,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        num_stages=config['model']['num_stages']
    ).to(device)

    gnn_weights_path = "models_weights/temporal_gnn_world_model.pt"
    if os.path.exists(gnn_weights_path):
        gnn_model.load_state_dict(torch.load(gnn_weights_path, map_location=device))
        gnn_model.eval()

    from preprocessing.node_feature_scaler import NodeFeatureScaler
    node_scaler = NodeFeatureScaler()
    node_scaler_path = "models_weights/node_feature_scaler.joblib"
    if os.path.exists(node_scaler_path):
        node_scaler.load(node_scaler_path)

    scaler = StateScaler()
    scaler_path = config['model']['scaler_path']
    if os.path.exists(scaler_path):
        scaler.load(scaler_path)

    return lstm_model, gnn_model, node_scaler, scaler, device

def main():
    config = load_config()

    # Sidebar Header Branding
    st.sidebar.title("🛡️ CyberForecaster")
    st.sidebar.caption("AI-Based Network Attack Forecasting System")
    st.sidebar.markdown("---")

    # 1. INFERENCE ENGINE SECTION
    st.sidebar.markdown('<div class="sidebar-section-header">🤖 INFERENCE ENGINE</div>', unsafe_allow_html=True)
    selected_model_name = st.sidebar.radio(
        "Select Active Engine",
        ["Baseline LSTM World Model", "Advanced Temporal GNN + LSTM"],
        index=1,
        label_visibility="collapsed"
    )

    # Compact Model Status Display
    st.sidebar.markdown(f"""
    <div class="file-status-box" style="border-left: 3px solid #38BDF8;">
        <div><b>MODEL STATUS:</b> <span style="color:#4ADE80;">● Loaded</span></div>
        <div><b>MODEL TYPE:</b> {selected_model_name}</div>
        <div><b>MODE:</b> Forward Simulation</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # 2. DATA INGESTION SECTION
    st.sidebar.markdown('<div class="sidebar-section-header">📡 DATA INGESTION</div>', unsafe_allow_html=True)
    data_source = st.sidebar.radio(
        "Select Input Mode",
        ["Demo Dataset (CSV)", "Upload CSV", "Upload PCAP"],
        label_visibility="collapsed"
    )

    df = None
    upload_status_msg = ""
    if data_source == "Demo Dataset (CSV)":
        demo_path = config['data']['demo_csv_path']
        if not os.path.exists(demo_path):
            st.sidebar.warning("Demo CSV not found. Generating...")
            from data.demo_generator import generate_demo_dataset
            generate_demo_dataset(demo_path)
        df = load_flow_csv(demo_path)
        upload_status_msg = f"✓ Demo CSV loaded<br>✓ {len(df):,} flows parsed<br>✓ Ready for inference"
        st.sidebar.markdown(f'<div class="file-status-box">{upload_status_msg}</div>', unsafe_allow_html=True)

    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload Flow CSV File", type=["csv"])
        if uploaded_file is not None:
            df = load_flow_csv(uploaded_file)
            upload_status_msg = f"✓ CSV loaded<br>✓ {len(df):,} flows parsed<br>✓ Ready for inference"
            st.sidebar.markdown(f'<div class="file-status-box" style="border-left: 3px solid #22C55E;">{upload_status_msg}</div>', unsafe_allow_html=True)
        else:
            st.sidebar.info("Please upload a flow CSV file.")

    elif data_source == "Upload PCAP":
        uploaded_pcap = st.sidebar.file_uploader("Upload PCAP / PCAPNG File", type=["pcap", "pcapng"])
        if uploaded_pcap is not None:
            temp_pcap_path = "data/demo/uploaded_temp.pcap"
            os.makedirs("data/demo", exist_ok=True)
            with open(temp_pcap_path, "wb") as f:
                f.write(uploaded_pcap.read())
            df = parse_pcap_file(temp_pcap_path)
            upload_status_msg = f"✓ PCAP parsed<br>✓ {len(df):,} flows extracted<br>✓ Ready for inference"
            st.sidebar.markdown(f'<div class="file-status-box" style="border-left: 3px solid #22C55E;">{upload_status_msg}</div>', unsafe_allow_html=True)
        else:
            st.sidebar.info("Please upload a PCAP packet capture file.")

    if df is None or df.empty:
        st.error("No dataset loaded. Please select a valid data source or upload telemetry data.")
        return

    # Load Models and Scalers
    lstm_model, gnn_model, node_scaler, scaler, device = load_all_models(config)

    st.sidebar.markdown("---")

    # 3. TIME CONTROLS SECTION
    st.sidebar.markdown('<div class="sidebar-section-header">⏱️ TIME CONTROLS</div>', unsafe_allow_html=True)
    window_sec = st.sidebar.slider("Time Window (seconds)", 2, 15, config['data']['window_seconds'])
    seq_len = config['sequence']['sequence_length']

    states = build_network_states(df, window_seconds=window_sec)
    if len(states) <= seq_len:
        st.error(f"Insufficient time windows ({len(states)}) for sequence length {seq_len}.")
        return

    X, y_state, y_attack, y_stage, timestamps = create_sequences(states, sequence_length=seq_len)
    
    selected_idx = st.sidebar.slider("SOC Window Index t", 0, len(X) - 1, len(X) - 1)

    current_seq_orig = X[selected_idx]
    current_timestamp = timestamps[selected_idx]
    current_df_windows = [states[j].get('df_window', pd.DataFrame()) for j in range(selected_idx, selected_idx + seq_len)]

    k_horizon = st.sidebar.slider("Forecast Horizon K", 1, 10, config['sequence']['forecast_horizon'])

    # Perform Model Rollout based on Selected Model
    if selected_model_name == "Baseline LSTM World Model":
        rollout_results = perform_k_step_rollout(lstm_model, scaler, current_seq_orig, k_steps=k_horizon, device=device)
        active_model_obj = lstm_model
    else:
        rollout_results = perform_gnn_k_step_rollout(
            gnn_model, scaler, node_scaler, current_seq_orig, historical_df_windows=current_df_windows, k_steps=k_horizon, device=device
        )
        active_model_obj = gnn_model

    # Risk Analysis
    risk_info = calculate_network_risk(
        rollout_results,
        warning_threshold=config['forecasting']['warning_threshold'],
        critical_threshold=config['forecasting']['critical_threshold']
    )

    # Lightweight Drift Detection
    drift_detector = DistributionDriftDetector(X[:int(len(X)*0.7)])
    drift_info = drift_detector.detect_drift(current_seq_orig)

    st.sidebar.markdown("---")

    # 4. NAVIGATION SECTION
    st.sidebar.markdown('<div class="sidebar-section-header">🧭 NAVIGATION</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "1. Overview",
            "2. Attack Forecast",
            "3. Attack Progression",
            "4. Network Graph Topology",
            "5. Explainability",
            "6. Traffic Explorer",
            "7. Model Performance"
        ],
        label_visibility="collapsed"
    )

    # ----------------------------------------------------
    # SOC TOP HEADER BAR
    # ----------------------------------------------------
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #111722 0%, #1A2332 100%); padding: 18px 24px; border-radius: 10px; border-left: 5px solid #38BDF8; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        <div>
            <h1 style="margin: 0; color: #FFFFFF; font-size: 24px; font-weight: 700; letter-spacing: 0.5px;">CyberForecaster SOC</h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 13px;">AI-Based Predictive Network Defence</p>
        </div>
        <div style="text-align: right;">
            <span class="status-badge-online">● SYSTEM ONLINE</span>
            <div style="margin-top: 6px; color: #38BDF8; font-size: 12px; font-weight: 600;">Engine: {selected_model_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Distribution Drift Warning Banner
    if drift_info['drift_warning']:
        st.markdown(f"""
        <div style="background-color: #2D1A00; border-left: 4px solid #F97316; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; color: #FDBA74; font-size: 13px;">
            ⚠️ <b>DISTRIBUTION DRIFT DETECTED:</b> {drift_info['warning_message']}
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # PAGE 1: OVERVIEW
    # ----------------------------------------------------
    if page == "1. Overview":
        st.subheader("🌐 Network Threat Overview")

        # Determine Threat Card Style Class
        tl_str = risk_info['threat_level']
        if tl_str == "CRITICAL":
            threat_class = "threat-critical"
            threat_color = "#EF4444"
        elif tl_str == "HIGH WARNING":
            threat_class = "threat-warning"
            threat_color = "#F97316"
        elif tl_str == "ELEVATED":
            threat_class = "threat-elevated"
            threat_color = "#EAB308"
        else:
            threat_class = "threat-normal"
            threat_color = "#22C55E"

        # TOP THREAT OVERVIEW KPI CARDS
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card {threat_class}">
                <div class="kpi-label">THREAT LEVEL</div>
                <div class="kpi-value" style="color: {threat_color};">{tl_str}</div>
                <div class="kpi-subtext">Current window index t</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">ATTACK PROBABILITY</div>
                <div class="kpi-value">{risk_info['max_future_risk']*100:.1f}%</div>
                <div class="kpi-subtext">Max rollout step risk</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">PREDICTED PEAK STAGE</div>
                <div class="kpi-value" style="color: #38BDF8;">{risk_info['predicted_peak_stage']}</div>
                <div class="kpi-subtext">ATT&CK Stage Head</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">ACTIVE MODEL</div>
                <div class="kpi-value" style="font-size: 16px; margin-top: 4px;">{selected_model_name}</div>
                <div class="kpi-subtext">Neural Transition Engine</div>
            </div>
            """, unsafe_allow_html=True)

        if risk_info['alert_triggered']:
            st.error(f"⚠️ **SOC ALERT TRIGGERED:** Future attack probability ({risk_info['max_future_risk']*100:.1f}%) exceeds warning threshold ({config['forecasting']['warning_threshold']*100:.0f}%). Forecasted Peak Stage: **{risk_info['predicted_peak_stage']}**")

        st.markdown("---")
        st.subheader("📈 Multi-Step Forward Horizon Preview")
        
        # Interactive Probability Rollout Bar/Scatter Chart
        horizon_steps = [r['horizon_step'] for r in rollout_results]
        horizon_probs = [r['attack_probability'] * 100 for r in rollout_results]
        horizon_stages = [STAGE_DESCRIPTIONS[r['predicted_stage_id']]['name'] for r in rollout_results]

        fig_hor = go.Figure()
        fig_hor.add_trace(go.Bar(
            x=horizon_steps,
            y=horizon_probs,
            text=[f"{p:.1f}%<br>({s})" for p, s in zip(horizon_probs, horizon_stages)],
            textposition='auto',
            marker_color=['#EF4444' if p >= 85 else ('#F97316' if p >= 70 else '#38BDF8') for p in horizon_probs],
            name="Attack Risk (%)"
        ))

        fig_hor.add_hline(y=config['forecasting']['warning_threshold']*100, line_dash="dot", line_color="#F97316", annotation_text="Warning (70%)")
        fig_hor.add_hline(y=config['forecasting']['critical_threshold']*100, line_dash="dot", line_color="#EF4444", annotation_text="Critical (85%)")

        fig_hor.update_layout(
            title=f"Forecast Horizon Risk Rollout S(t+1)...S(t+{k_horizon})",
            xaxis_title="Forward Time Step",
            yaxis_title="Predicted Attack Probability (%)",
            yaxis=dict(range=[0, 105]),
            template="plotly_dark",
            height=340,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_hor, use_container_width=True)

        horizon_df = pd.DataFrame([
            {
                "Horizon Step": r['horizon_step'],
                "Attack Probability (%)": f"{r['attack_probability'] * 100:.2f}%",
                "Predicted Stage": STAGE_DESCRIPTIONS[r['predicted_stage_id']]['name'],
                "SYN Ratio": f"{r['state_dict']['syn_ratio']:.4f}",
                "Port Entropy": f"{r['state_dict']['port_entropy']:.4f}",
                "Total Bytes": f"{r['state_dict']['total_bytes']:,.0f}"
            } for r in rollout_results
        ])
        
        st.dataframe(horizon_df, use_container_width=True)

    # ----------------------------------------------------
    # PAGE 2: ATTACK FORECAST
    # ----------------------------------------------------
    elif page == "2. Attack Forecast":
        st.subheader("🔮 Multi-Step Forward Attack Forecast")

        scaled_all_X = scaler.transform(X)
        X_tensor_all = torch.tensor(scaled_all_X, dtype=torch.float32).to(device)

        with torch.no_grad():
            if selected_model_name == "Baseline LSTM World Model":
                _, p_att_all, _ = lstm_model(X_tensor_all)
            else:
                _, p_att_all, _ = gnn_model(X_tensor_all)
            all_probs = p_att_all.cpu().numpy().flatten()

        fig = go.Figure()
        ts_str = [str(t).split('.')[0] for t in timestamps[:selected_idx+1]]
        fig.add_trace(go.Scatter(
            x=ts_str,
            y=all_probs[:selected_idx+1] * 100,
            mode='lines+markers',
            name='Observed Network Risk (%)',
            line=dict(color='#38BDF8', width=3)
        ))

        future_ts = [f"t+{r['step_index']} ({r['horizon_step']})" for r in rollout_results]
        future_probs = [r['attack_probability'] * 100 for r in rollout_results]

        fig.add_trace(go.Scatter(
            x=[ts_str[-1]] + future_ts,
            y=[all_probs[selected_idx] * 100] + future_probs,
            mode='lines+markers',
            name=f'Forecasted Rollout ({selected_model_name})',
            line=dict(color='#EF4444', width=3, dash='dash')
        ))

        fig.add_hline(y=config['forecasting']['warning_threshold']*100, line_dash="dot", line_color="#F97316", annotation_text="Warning Threshold (70%)")
        fig.add_hline(y=config['forecasting']['critical_threshold']*100, line_dash="dot", line_color="#EF4444", annotation_text="Critical Threshold (85%)")

        fig.update_layout(
            title=f"Temporal Network Attack Probability Timeline ({selected_model_name})",
            xaxis_title="Timeline Window",
            yaxis_title="Attack Probability (%)",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # PAGE 3: ATTACK PROGRESSION
    # ----------------------------------------------------
    elif page == "3. Attack Progression":
        st.subheader("🎯 MITRE ATT&CK Stage Progression Mapping")

        current_stage_id = rollout_results[0]['predicted_stage_id']
        current_stage_info = get_stage_details(current_stage_id)

        st.info(f"**Current Forecasted Stage:** Stage {current_stage_id} - **{current_stage_info['name']}** (MITRE ID: {current_stage_info['mitre_id']})\n\n{current_stage_info['description']}")

        st.markdown("### Stage Pipeline")
        cols = st.columns(6)
        for s_id in range(6):
            s_data = get_stage_details(s_id)
            with cols[s_id]:
                if s_id == current_stage_id:
                    st.markdown(f"""
                    <div style="background: #064E3B; border: 2px solid #10B981; border-radius: 8px; padding: 12px; text-align: center;">
                        <div style="font-size: 11px; color: #6EE7B7; font-weight: 700;">STAGE {s_id} (ACTIVE)</div>
                        <div style="font-size: 14px; color: #FFFFFF; font-weight: 700; margin-top: 4px;">{s_data['name']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif s_id < current_stage_id:
                    st.markdown(f"""
                    <div style="background: #151C28; border: 1px solid #334155; border-radius: 8px; padding: 12px; text-align: center;">
                        <div style="font-size: 11px; color: #94A3B8;">STAGE {s_id}</div>
                        <div style="font-size: 13px; color: #CBD5E1; margin-top: 4px;">✓ {s_data['name']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #0F141C; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; text-align: center; opacity: 0.6;">
                        <div style="font-size: 11px; color: #64748B;">STAGE {s_id}</div>
                        <div style="font-size: 13px; color: #64748B; margin-top: 4px;">{s_data['name']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔍 Stage Indicators & Key Telemetry Markers")
        for key_ind in current_stage_info['key_indicators']:
            st.markdown(f"- 🚩 **{key_ind}**")

    # ----------------------------------------------------
    # PAGE 4: NETWORK GRAPH TOPOLOGY
    # ----------------------------------------------------
    elif page == "4. Network Graph Topology":
        st.subheader("NETWORK COMMUNICATION GRAPH")
        st.caption("Visualizing host nodes (IPs), directed communication edges, and structural node degree anomalies.")

        df_sorted = df.sort_values(by='Timestamp').set_index('Timestamp')
        grouped_win = df_sorted.resample(f"{int(window_sec)}s")
        win_dfs = [grp for _, grp in grouped_win]

        curr_win_df = win_dfs[selected_idx] if selected_idx < len(win_dfs) else pd.DataFrame()
        g_dict = build_window_graph(curr_win_df, window_seconds=window_sec)
        g_analysis = explain_network_graph(g_dict)

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.metric("Total Endpoints (Nodes)", g_analysis['total_nodes'])
        with col_g2:
            st.metric("Active Connections (Edges)", g_analysis['total_edges'])
        with col_g3:
            st.metric("High-Risk Host Endpoints", len(g_analysis['high_risk_nodes']))

        st.markdown("""
        <div style="background: #151C28; padding: 10px 16px; border-radius: 6px; border: 1px solid #232D3F; margin-bottom: 12px; display: flex; gap: 20px; font-size: 13px;">
            <div><span style="color:#38BDF8;">●</span> <b>Internal Host</b> (Normal Node)</div>
            <div><span style="color:#EF4444;">●</span> <b>Suspicious Host</b> (High-Risk Anomaly)</div>
            <div><span style="color:#64748B;">──</span> <b>Communication</b> (Flow Edge)</div>
        </div>
        """, unsafe_allow_html=True)

        # Construct NetworkX Graph for Plotly rendering
        G = nx.DiGraph()
        for ip in g_dict['ip_list']:
            G.add_node(ip)

        edge_idx = g_dict['edge_index']
        if edge_idx.ndim == 2 and edge_idx.shape[1] > 0:
            for e_i in range(edge_idx.shape[1]):
                u_ip = g_dict['ip_list'][edge_idx[0, e_i]]
                v_ip = g_dict['ip_list'][edge_idx[1, e_i]]
                G.add_edge(u_ip, v_ip)

        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#475569'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []

        high_risk_ips = set(n['ip'] for n in g_analysis['high_risk_nodes'])

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Host IP: {node}")
            if node in high_risk_ips:
                node_color.append('#EF4444')
            else:
                node_color.append('#38BDF8')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[f"  {n}" for n in G.nodes()],
            textposition="top center",
            marker=dict(
                color=node_color,
                size=16,
                line_width=2
            )
        )

        fig_graph = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='Dynamic Network Host Graph G(t)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        template="plotly_dark",
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=550
                    ))

        st.plotly_chart(fig_graph, use_container_width=True)

    # ----------------------------------------------------
    # PAGE 5: EXPLAINABILITY
    # ----------------------------------------------------
    elif page == "5. Explainability":
        st.subheader("WHY IS THE MODEL CONCERNED?")
        st.caption("Feature Attribution & Gradient Saliency Breakdown")

        from explainability.shap_explainer import GradientSaliencyExplainer
        explainer = GradientSaliencyExplainer(active_model_obj, scaler)
        explanation = explainer.explain_instance(current_seq_orig)

        st.info(f"ℹ️ **Explanation Method:** {explanation['method']}\n\n*Displaying real-time gradient saliency attributions for the current window prediction.*")

        top_feats = explanation['top_features']
        top_df = pd.DataFrame(top_feats)

        fig_exp = px.bar(
            top_df,
            x='attribution',
            y='feature',
            orientation='h',
            title=f"Top 10 Feature Contributions to Predicted Attack Risk [{selected_model_name}]",
            labels={'attribution': 'Attribution Weight', 'feature': 'Network State Feature'},
            color='attribution',
            color_continuous_scale=px.colors.sequential.Teal,
            template="plotly_dark"
        )
        fig_exp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_exp, use_container_width=True)

        st.markdown("### Detailed Feature Attribution Table")
        st.dataframe(top_df[['feature', 'attribution', 'original_value', 'scaled_value']], use_container_width=True)

    # ----------------------------------------------------
    # PAGE 6: TRAFFIC EXPLORER
    # ----------------------------------------------------
    elif page == "6. Traffic Explorer":
        st.subheader("🔍 Flow Telemetry Explorer")

        col_search, col_filter = st.columns(2)
        with col_search:
            search_query = st.text_input("Search IP / Port", "")
        with col_filter:
            proto_filter = st.selectbox("Protocol Filter", ["All", "TCP (6)", "UDP (17)"])

        filtered_df = df.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df['Src_IP'].astype(str).str.contains(search_query) |
                filtered_df['Dst_IP'].astype(str).str.contains(search_query) |
                filtered_df['Dst_Port'].astype(str).str.contains(search_query)
            ]

        if proto_filter == "TCP (6)":
            filtered_df = filtered_df[filtered_df['Protocol'] == 6]
        elif proto_filter == "UDP (17)":
            filtered_df = filtered_df[filtered_df['Protocol'] == 17]

        st.dataframe(filtered_df, use_container_width=True, height=450)

    # ----------------------------------------------------
    # PAGE 7: MODEL PERFORMANCE & COMPARISON
    # ----------------------------------------------------
    elif page == "7. Model Performance":
        st.subheader("📊 Comparative Benchmark Evaluation (4 Models)")

        comp_path = "experiments/results/model_comparison.json"
        if not os.path.exists(comp_path):
            comp_path = "experiments/results/demo_results.json"

        if os.path.exists(comp_path):
            with open(comp_path, "r") as f:
                comp_data = json.load(f)

            meta = comp_data.get('evaluation_metadata', {})
            m1 = comp_data.get('Model_1_Static_Logistic_Regression', comp_data.get('Model_1_Logistic_Regression_Baseline', {}))
            m2 = comp_data.get('Model_2_Temporal_Logistic_Regression', {})
            m3 = comp_data.get('Model_3_Temporal_LSTM_WorldModel', comp_data.get('Model_2_Temporal_LSTM_WorldModel', {}))
            m4 = comp_data.get('Model_4_Temporal_GNN_WorldModel', comp_data.get('Model_3_Temporal_GNN_WorldModel', {}))

            st.info(f"**Evaluated Dataset:** {meta.get('dataset')}\n\n**Test Samples:** {meta.get('test_samples')} ({meta.get('benign_test_samples')} Benign / {meta.get('attack_test_samples')} Attack)")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("### 🔹 Baseline A: Static LR")
                st.write(f"- **Precision:** {m1.get('Precision')}")
                st.write(f"- **Recall:** {m1.get('Recall')}")
                st.write(f"- **F1-Score:** {m1.get('F1_Score')}")
                st.write(f"- **FPR:** {m1.get('FPR')}")

            with col2:
                st.markdown("### 📈 Baseline B: Temporal LR")
                st.write(f"- **Precision:** {m2.get('Precision')}")
                st.write(f"- **Recall:** {m2.get('Recall')}")
                st.write(f"- **F1-Score:** {m2.get('F1_Score')}")
                st.write(f"- **FPR:** {m2.get('FPR')}")

            with col3:
                st.markdown("### 🏆 Proposed: Temporal LSTM")
                st.write(f"- **Precision:** {m3.get('Precision')}")
                st.write(f"- **Recall:** {m3.get('Recall')}")
                st.write(f"- **F1-Score:** {m3.get('F1_Score')}")
                st.write(f"- **FPR:** {m3.get('FPR')}")
                st.write(f"- **Next-State RMSE:** {m3.get('NextState_RMSE')}")

            with col4:
                st.markdown("### 🕸️ Experimental: Temporal GNN")
                st.write(f"- **Precision:** {m4.get('Precision')}")
                st.write(f"- **Recall:** {m4.get('Recall')}")
                st.write(f"- **F1-Score:** {m4.get('F1_Score')}")
                st.write(f"- **FPR:** {m4.get('FPR')}")
                st.write(f"- **Next-State RMSE:** {m4.get('NextState_RMSE')}")

            st.markdown("---")
            st.markdown("### Side-by-Side 4-Model Comparison Matrix")
            comp_table = pd.DataFrame([
                {"Model": "Baseline A (Static LR S(t))", **m1},
                {"Model": "Baseline B (Temporal LR S(t-9)...S(t))", **m2},
                {"Model": "Proposed (Temporal LSTM World Model)", **m3},
                {"Model": "Experimental (Temporal GNN + LSTM)", **m4}
            ])
            st.dataframe(comp_table, use_container_width=True)
        else:
            st.info("Run model comparison script (`python training/compare_models.py`) to generate comparative benchmarks.")

if __name__ == "__main__":
    main()
