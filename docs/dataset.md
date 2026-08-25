# Dataset Specification & Adapters: CyberForecaster

## 1. Supported Cybersecurity Datasets

CyberForecaster is designed with dataset-agnostic feature adapters supporting:

1. **CIC-IDS2018 / CIC-IDS2017:** Flow-level CSV files exported by CICFlowMeter.
2. **UNSW-NB15:** Network flow dataset with labeled attack categories.
3. **CTU-13:** Botnet traffic capture dataset.
4. **PCAP Files:** Raw packet captures parsed via Scapy / dpkt.

---

## 2. Feature Schema S(t) (23 Dimensions)

| Index | Feature Name | Description |
|---|---|---|
| 0 | `total_packets` | Aggregated packet count in window |
| 1 | `total_bytes` | Aggregated byte volume in window |
| 2 | `unique_src_ips` | Count of distinct source IPs |
| 3 | `unique_dst_ips` | Count of distinct destination IPs |
| 4 | `unique_dst_ports` | Count of distinct destination ports |
| 5 | `tcp_ratio` | Ratio of TCP flows |
| 6 | `udp_ratio` | Ratio of UDP flows |
| 7 | `syn_ratio` | SYN flag count / total packets |
| 8 | `ack_ratio` | ACK flag count / total packets |
| 9 | `rst_ratio` | RST flag count / total packets |
| 10 | `fin_ratio` | FIN flag count / total packets |
| 11 | `mean_packet_size` | Average packet size in bytes |
| 12 | `packet_size_variance` | Variance of packet size |
| 13 | `mean_IAT` | Mean packet Inter-Arrival Time |
| 14 | `IAT_variance` | Variance of Inter-Arrival Time |
| 15 | `max_IAT` | Maximum Inter-Arrival Time |
| 16 | `retransmission_rate` | Connection reset/retransmission rate |
| 17 | `ttl_mean` | Mean IP Time-To-Live |
| 18 | `ttl_variance` | Variance of IP TTL |
| 19 | `inbound_outbound_ratio` | Ratio of inbound to outbound flows |
| 20 | `failed_connection_rate` | Proportion of failed/reset connections |
| 21 | `port_entropy` | Shannon entropy of destination ports |
| 22 | `connection_rate` | Connection flow rate per second |

---

## 3. Demo Dataset Generation

The project includes a self-contained dataset generator (`data/demo_generator.py`) producing:
- `data/demo/demo_cicids2018.csv`: 2400 flow records across a 600-second timeline with 6 attack stages.
- `data/demo/demo_sample.pcap`: Valid synthetic packet capture file for testing PCAP parsing.
