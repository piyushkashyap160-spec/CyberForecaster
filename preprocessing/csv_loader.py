import os
import io
import pandas as pd
import numpy as np

COLUMN_MAPPINGS = {
    # Timestamps
    'timestamp': 'Timestamp',
    'Time': 'Timestamp',
    'frame.time': 'Timestamp',

    # IPs and Ports
    'src_ip': 'Src_IP',
    'Source IP': 'Src_IP',
    'src_host': 'Src_IP',

    'dst_ip': 'Dst_IP',
    'Destination IP': 'Dst_IP',
    'dst_host': 'Dst_IP',

    'src_port': 'Src_Port',
    'Source Port': 'Src_Port',

    'dst_port': 'Dst_Port',
    'Destination Port': 'Dst_Port',
    'Dst Port': 'Dst_Port',

    'protocol': 'Protocol',

    # Flow stats
    'flow_duration': 'Flow_Duration',
    'Flow Duration': 'Flow_Duration',

    'tot_pkts': 'Tot_Pkts',
    'Total Fwd Packets': 'Tot_Pkts',
    'Tot Fwd Pkts': 'Tot_Pkts',

    'tot_bytes': 'Tot_Bytes',
    'Total Length of Fwd Packets': 'Tot_Bytes',
    'TotLen Fwd Pkts': 'Tot_Bytes',

    # Flags
    'SYN Flag Count': 'SYN_Cnt',
    'syn_cnt': 'SYN_Cnt',
    'ACK Flag Count': 'ACK_Cnt',
    'ack_cnt': 'ACK_Cnt',
    'RST Flag Count': 'RST_Cnt',
    'rst_cnt': 'RST_Cnt',
    'FIN Flag Count': 'FIN_Cnt',
    'fin_cnt': 'FIN_Cnt',
    'PSH Flag Count': 'PSH_Cnt',
    'psh_cnt': 'PSH_Cnt',
    'URG Flag Count': 'URG_Cnt',
    'urg_cnt': 'URG_Cnt',

    # Rates & IAT
    'Flow Bytes/s': 'Bytes_Per_Sec',
    'Flow Packets/s': 'Pkts_Per_Sec',
    'Flow IAT Mean': 'Mean_IAT',
    'Flow IAT Std': 'Var_IAT',
    'Flow IAT Max': 'Max_IAT',
    'Pkt Size Avg': 'Mean_Pkt_Size',
    'Packet Length Mean': 'Mean_Pkt_Size',
    'Packet Length Std': 'Var_Pkt_Size',

    # Label
    'label': 'Label',
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

    # Rename mapped columns
    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_MAPPINGS:
            rename_dict[col] = COLUMN_MAPPINGS[col]
    
    df = df.rename(columns=rename_dict)

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
