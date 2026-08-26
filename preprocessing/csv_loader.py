import os
import io
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

COLUMN_MAPPINGS = {
    # Timestamps
    'timestamp': 'Timestamp',
    'Time': 'Timestamp',
    'frame.time': 'Timestamp',
    'Timestamp': 'Timestamp',

    # IPs and Ports
    'src_ip': 'Src_IP',
    'Source IP': 'Src_IP',
    'src_host': 'Src_IP',
    'Src IP': 'Src_IP',

    'dst_ip': 'Dst_IP',
    'Destination IP': 'Dst_IP',
    'dst_host': 'Dst_IP',
    'Dst IP': 'Dst_IP',

    'src_port': 'Src_Port',
    'Source Port': 'Src_Port',
    'Src Port': 'Src_Port',

    'dst_port': 'Dst_Port',
    'Destination Port': 'Dst_Port',
    'Dst Port': 'Dst_Port',

    'protocol': 'Protocol',
    'Protocol': 'Protocol',

    # Flow stats
    'flow_duration': 'Flow_Duration',
    'Flow Duration': 'Flow_Duration',

    'tot_pkts': 'Tot_Pkts',
    'Total Fwd Packets': 'Tot_Pkts',

    'tot_bytes': 'Tot_Bytes',
    'Total Length of Fwd Packets': 'Tot_Bytes',

    # Flags
    'SYN Flag Count': 'SYN_Cnt',
    'SYN Flag Cnt': 'SYN_Cnt',
    'syn_cnt': 'SYN_Cnt',

    'ACK Flag Count': 'ACK_Cnt',
    'ACK Flag Cnt': 'ACK_Cnt',
    'ack_cnt': 'ACK_Cnt',

    'RST Flag Count': 'RST_Cnt',
    'RST Flag Cnt': 'RST_Cnt',
    'rst_cnt': 'RST_Cnt',

    'FIN Flag Count': 'FIN_Cnt',
    'FIN Flag Cnt': 'FIN_Cnt',
    'fin_cnt': 'FIN_Cnt',

    'PSH Flag Count': 'PSH_Cnt',
    'PSH Flag Cnt': 'PSH_Cnt',
    'psh_cnt': 'PSH_Cnt',

    'URG Flag Count': 'URG_Cnt',
    'URG Flag Cnt': 'URG_Cnt',
    'urg_cnt': 'URG_Cnt',

    # Rates & IAT
    'Flow Bytes/s': 'Bytes_Per_Sec',
    'Flow Byts/s': 'Bytes_Per_Sec',
    'Flow Packets/s': 'Pkts_Per_Sec',
    'Flow Pkts/s': 'Pkts_Per_Sec',

    'Flow IAT Mean': 'Mean_IAT',
    'Flow IAT Max': 'Max_IAT',

    'Pkt Size Avg': 'Mean_Pkt_Size',
    'Pkt Len Mean': 'Mean_Pkt_Size',
    'Packet Length Mean': 'Mean_Pkt_Size',

    'Pkt Len Var': 'Var_Pkt_Size',

    # TTL mappings (present in PCAPs/custom tools, absent in default CICFlowMeter CSVs)
    'ttl_mean': 'TTL_Mean',
    'TTL_Mean': 'TTL_Mean',
    'ttl_var': 'TTL_Var',
    'TTL_Var': 'TTL_Var',

    # Label
    'label': 'Label',
    'Label': 'Label',
    'Stage': 'Stage'
}

def load_flow_csv(file_input) -> pd.DataFrame:
    """
    Loads flow-level dataset CSV and normalizes column headers across CIC-IDS datasets.
    Accepts:
      - Filepath string or os.PathLike
      - File-like buffer objects (Streamlit UploadedFile, io.BytesIO, io.StringIO)
    """
    if isinstance(file_input, (str, os.PathLike)):
        if not os.path.exists(file_input):
            raise FileNotFoundError(f"Dataset CSV not found at path: {file_input}")
        df = pd.read_csv(file_input)
    elif hasattr(file_input, 'read') or isinstance(file_input, (io.BytesIO, io.StringIO, io.IOBase)):
        df = pd.read_csv(file_input)
    else:
        raise TypeError(f"Expected file path string or file-like buffer, got {type(file_input)}")

    # Clean whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]

    # Explicit Feature Derivations for CICFlowMeter / CIC-IDS2018 datasets
    if 'Tot Fwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
        df['Tot_Pkts'] = pd.to_numeric(df['Tot Fwd Pkts'], errors='coerce').fillna(0) + pd.to_numeric(df['Tot Bwd Pkts'], errors='coerce').fillna(0)

    if 'TotLen Fwd Pkts' in df.columns and 'TotLen Bwd Pkts' in df.columns:
        df['Tot_Bytes'] = pd.to_numeric(df['TotLen Fwd Pkts'], errors='coerce').fillna(0) + pd.to_numeric(df['TotLen Bwd Pkts'], errors='coerce').fillna(0)

    if 'Flow IAT Std' in df.columns and 'Var_IAT' not in df.columns:
        flow_iat_std = pd.to_numeric(df['Flow IAT Std'], errors='coerce').fillna(0)
        df['Var_IAT'] = flow_iat_std ** 2

    if 'Pkt Len Var' in df.columns and 'Var_Pkt_Size' not in df.columns:
        df['Var_Pkt_Size'] = pd.to_numeric(df['Pkt Len Var'], errors='coerce').fillna(0)

    # Detect TTL feature availability
    has_ttl = any(col in df.columns for col in ['TTL_Mean', 'ttl_mean', 'TTL_Var', 'ttl_var', 'TTL', 'ttl'])
    if not has_ttl:
        logging.warning("TTL features ('TTL_Mean', 'TTL_Var') are absent from loaded CSV data source. Using default fallbacks (ttl_mean=64.0, ttl_variance=0.0). TTL features are PCAP-only.")

    # Rename mapped columns safely without creating duplicate columns
    rename_dict = {}
    assigned_targets = set()
    for col in df.columns:
        if col in COLUMN_MAPPINGS:
            target_name = COLUMN_MAPPINGS[col]
            if target_name not in assigned_targets:
                rename_dict[col] = target_name
                assigned_targets.add(target_name)
    
    df = df.rename(columns=rename_dict)
    df = df.loc[:, ~df.columns.duplicated()]

    # Convert Timestamp to pandas datetime if string
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        # Drop rows with invalid timestamps
        df = df.dropna(subset=['Timestamp'])
        df = df.sort_values(by='Timestamp').reset_index(drop=True)

    # Fill NaNs in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    # Replace infinity values
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)

    return df

