import os
from pathlib import Path
from typing import Any

import numpy as np
import wfdb


def read_ecg_record(
    base_dir: str | Path,
    record_path: str,
    source: str = "mimic_ecg"
) -> tuple[np.ndarray, int, list[str], dict[str, Any]]:
    """
    Unified entry point for reading an ECG record (MIMIC-IV-ECG or PTB-XL).
    
    Args:
        base_dir: The root directory for the dataset (e.g., path to mimic-iv-ecg-demo).
        record_path: The relative path to the record (e.g., "files/p10001217/s102172660/102172660").
        source: Dataset identifier constraint ("mimic_ecg" or "ptbxl"). Currently only supports wfdb compat.
        
    Returns:
        waveform: np.ndarray of shape (n_samples, n_leads) containing the signal (in physical units, usually mV)
        fs: int, sampling frequency 
        lead_names: list of str, names of the leads
        meta: dict, additional metadata from the head file
    """
    
    full_path = str(Path(base_dir) / record_path)
    
    # We use rdsamp to read the physical signal and its metadata
    # The physical signal is 2D numpy array: samples x leads
    record_data, meta = wfdb.rdsamp(full_path)
    
    fs = int(meta.get("fs", 500))
    lead_names = meta.get("sig_name", [])
    
    # Store standard fields explicitly in the meta dict just in case
    meta["source"] = source
    meta["record_path"] = record_path
    
    return record_data, fs, lead_names, meta

def read_ecg_header(
    base_dir: str | Path,
    record_path: str,
) -> dict[str, Any]:
    """
    Utility to read *only* the header file (fast, without loading signal data).
    Useful for building dataset manifests.
    """
    full_path = str(Path(base_dir) / record_path)
    header = wfdb.rdheader(full_path)
    
    return {
        "fs": int(header.fs),
        "n_samples": int(header.sig_len),
        "n_leads": int(header.n_sig),
        "lead_names": header.sig_name,
        "base_date": str(header.base_date) if header.base_date else None,
        "base_time": str(header.base_time) if header.base_time else None,
    }
