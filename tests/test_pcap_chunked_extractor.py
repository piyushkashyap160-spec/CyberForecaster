import os
import pytest
import numpy as np
from preprocessing.pcap_chunked_extractor import ChunkedPCAPExtractor, calculate_entropy

def test_entropy_calculation():
    counts_equal = [10, 10, 10, 10]
    ent = calculate_entropy(counts_equal)
    assert ent == pytest.approx(2.0, abs=1e-2)

    counts_zero = [0, 0]
    assert calculate_entropy(counts_zero) == 0.0

def test_chunked_pcap_extractor_zip_missing():
    extractor = ChunkedPCAPExtractor(window_seconds=5.0)
    records = extractor.extract_from_zip_stream("data/non_existent_zip.zip")
    assert records == []

def test_chunked_pcap_extractor_real_zip():
    zip_path = "data/real/cicids2018/pcap.zip"
    if not os.path.exists(zip_path):
        pytest.skip(f"{zip_path} not available for test")

    extractor = ChunkedPCAPExtractor(window_seconds=5.0)
    records = extractor.extract_from_zip_stream(zip_path, max_files=2, max_packets_per_file=1000)

    assert len(records) > 0
    rec = records[0]
    assert "pcap_ttl_mean" in rec
    assert "pcap_ttl_var" in rec
    assert "pcap_port_entropy" in rec
    assert rec["pcap_enriched_flag"] == 1.0
    assert rec["pcap_ttl_mean"] > 0
