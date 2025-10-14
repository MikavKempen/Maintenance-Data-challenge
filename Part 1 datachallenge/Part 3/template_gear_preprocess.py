"""
    1CM290 Maintenance Optimization and Engineering (Lecturer: J. Lee)
    Assignment: Data Challenges 2025
    Challenge: Detection of faults in gears.
    Completed preprocessing template.
    Uses only allowed packages: numpy, pandas, matplotlib, scipy.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert

# Read Raw Data
def read_raw_data(path):
    data_dict = {}  # Data of each sample is stored with key (i, h, v, n)
    for file in os.listdir(path):
        if not file.lower().endswith('.csv'):
            continue
        print("Loading:", file)  # Check the file name during execution
        key = file.split('.')[0].split('_')
        i = int(key[0])  # i: index of data
        h = int(key[1])  # h: label of data (0: normal, 1: fault)
        v = int(key[2].split('V')[1])  # v: rotational speed (RPM)
        n = int(key[3].split('N')[0])  # n: load (Nm)
        key = (i, h, v, n)
        # read CSV into DataFrame
        df = pd.read_csv(os.path.join(path, file), header=None)
        # If file has more than 3 columns, keep first 3 (X, Y, Z)
        data_dict[key] = df
    return data_dict


# load raw data folders (assumes these folders exist and contain csv files)
data_train_raw = read_raw_data('data_gear/train/')
data_valid_raw = read_raw_data('data_gear/valid/')
data_test_raw = read_raw_data('data_gear/test/')


# Task (a) Preprocess the raw data
def preprocess(data_dict, sample_rate=20480, n_bands=20, n_harmonics=5, normalize_signal=False):
    """
    Preprocess all samples in data_dict.
    Outputs a dict with same keys (i,h,v,n) and value:
      (raw_array, time_feature_vector, band_energy_vector, harmonic_vector)
    Where:
      - raw_array: (N,3) numpy array (DC removed, maybe normalized)
      - time_feature_vector: features extracted per axis concatenated
      - band_energy_vector: relative band energies (length n_bands)
      - harmonic_vector: normalized magnitudes at mesh frequency and harmonics
    """
    data_dict_fft = {}

    # gear parameters (driving gear tooth count provided in assignment)
    teeth_driving = 40

    for key in data_dict:
        # Unpack key
        i, h, v, n = key
        df = data_dict[key]
        arr = df.values
        # Keep first three columns as axes X,Y,Z (if file has only 3 columns this is fine)
        if arr.ndim == 1:
            # single column file — not expected but handle gracefully
            arr = arr.reshape((-1, 1))
        if arr.shape[1] >= 3:
            sig = arr[:, :3].astype(np.float64)
        else:
            # If file has fewer than 3 columns, pad with zeros (unlikely)
            tmp = np.zeros((arr.shape[0], 3), dtype=np.float64)
            tmp[:, :arr.shape[1]] = arr
            sig = tmp

        # Safety: ensure length is as expected (3 seconds * fs) or proceed anyway
        # Remove DC offset per axis
        sig = sig - np.mean(sig, axis=0, keepdims=True)

        # Optional normalization by std (per-sample normalization)
        if normalize_signal:
            stds = np.std(sig, axis=0, keepdims=True)
            stds[stds == 0] = 1.0
            sig = sig / stds

        # TIME-DOMAIN FEATURES (per axis)
        time_feats = []
        for ax in range(3):
            s = sig[:, ax]
            mean = np.mean(s)
            std = np.std(s)
            ptp = np.max(s) - np.min(s)  # peak-to-peak
            rms = np.sqrt(np.mean(s**2))
            # use pandas Series for skew/kurtosis (allowed package)
            s_series = pd.Series(s)
            kurt = float(s_series.kurtosis())
            skew = float(s_series.skew())
            # envelope RMS via analytic signal (useful for impact detection)
            analytic = hilbert(s)
            envelope = np.abs(analytic)
            env_rms = np.sqrt(np.mean(envelope**2))

            time_feats.extend([mean, std, ptp, rms, skew, kurt, env_rms])

        time_feats = np.array(time_feats, dtype=np.float64)  # length = 3 * 7 = 21

        # FREQUENCY-DOMAIN FEATURES
        N = sig.shape[0]
        # FFT per axis can be computed, but we'll use axis 0 (X) for band energies for simplicity
        xf = fft(sig[:, 0])
        freqs = fftfreq(N, 1.0 / sample_rate)
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        mag = np.abs(xf[pos_mask]) / N  # magnitude spectrum (normalized by N)

        # Band energies: split 0..Nyquist into n_bands and compute relative energy
        nyq = sample_rate / 2.0
        band_edges = np.linspace(0.0, nyq, n_bands + 1)
        band_energies = []
        for bi in range(n_bands):
            mask = (freqs_pos >= band_edges[bi]) & (freqs_pos < band_edges[bi + 1])
            # energy = sum of squared magnitudes
            be = np.sum((mag[mask]) ** 2)
            band_energies.append(be)
        band_energies = np.array(band_e
