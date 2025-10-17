# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
from sklearn.decomposition import PCA

# Task a

## FUNCTIONS

def plot_raw_data(raw_data_dict, key, duration=3, save_path=None):

    ### Getting raw DataFrame for this experiment
    df = raw_data_dict[key]

    ## Extracting each axis
    x_data = df['X']
    y_data = df['Y']
    z_data = df['Z']

    ### Setting time vector
    num_samples = len(x_data)
    t = np.linspace(0, duration, num_samples)

    ### Creating figure
    plt.figure(figsize=(12, 10))

    #### X axis
    plt.subplot(3, 1, 1)
    plt.plot(t, x_data, color='red')
    plt.title(f'Experiment {key[0]} – X axis (Raw)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration')
    plt.grid(True)

    #### Y axis
    plt.subplot(3, 1, 2)
    plt.plot(t, y_data, color='green')
    plt.title(f'Experiment {key[0]} – Y axis (Raw)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration')
    plt.grid(True)

    #### Z axis
    plt.subplot(3, 1, 3)
    plt.plot(t, z_data, color='blue')
    plt.title(f'Experiment {key[0]} – Z axis (Raw)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration')
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def read_raw_data(path, verbose=True):
    data_dict = {}
    files = sorted(os.listdir(path))  #sort to keep order consistent

    for idx, file in enumerate(files, 1):
        key_parts = file.split('.')[0].split('_')
        i = int(key_parts[0])
        h = int(key_parts[1])
        v = int(key_parts[2].split('V')[1])
        n = int(key_parts[3].split('N')[0])
        key = (i, h, v, n)
        data_dict[key] = pd.read_csv(os.path.join(path, file))

        # Print progress every 10 files
        if verbose and idx % 10 == 0:
            print(f"Loaded {idx}/{len(files)} files from {path}...")

    if verbose:
        print(f"Completed loading {len(files)} files from {path}.") # Print every complete folder

    return data_dict

def transform_fft(data_df, sample_rate=20480):

    ### Extract vibration signals
    x = data_df['X'].values
    y = data_df['Y'].values
    z = data_df['Z'].values

    ### Compute FFTs
    X_fft = fft(x)
    Y_fft = fft(y)
    Z_fft = fft(z)

    ### Frequency bins
    freqs = fftfreq(len(x), d=1/sample_rate)

    ### Keep only positive frequencies (real signals are symmetric)
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    X_fft = np.abs(X_fft[pos_mask]) / len(x)
    Y_fft = np.abs(Y_fft[pos_mask]) / len(y)
    Z_fft = np.abs(Z_fft[pos_mask]) / len(z)

    ### Return as dictionary
    return {
        'freqs': freqs,
        'X_fft': X_fft,
        'Y_fft': Y_fft,
        'Z_fft': Z_fft
    }

def apply_fft_to_dataset(data_dict, sample_rate=20480, verbose=True):

    data_dict_fft = {}
    total_files = len(data_dict)

    for idx, (key, df) in enumerate(data_dict.items(), 1):
        data_dict_fft[key] = transform_fft(df, sample_rate)

        ### Print progress every 10 files
        if verbose and idx % 10 == 0:
            print(f"Transformed {idx}/{total_files} files...")

    if verbose:
        print(f"Completed FFT transformation for {total_files} files.") # Print every complete folder

    return data_dict_fft

def normalize_fft_dataset(data_dict, mean=None, std=None, verbose=True):
    all_features = []
    keys_order = []

    for key, fft_data in data_dict.items():
        feature_vector = np.hstack([fft_data['X_fft'],
                                    fft_data['Y_fft'],
                                    fft_data['Z_fft']])
        all_features.append(feature_vector)
        keys_order.append(key)

    all_features = np.vstack(all_features)  # shape: (num_experiments, num_features)

    ### Computing mean & std
    if mean is None:
        mean = all_features.mean(axis=0)
    if std is None:
        std = all_features.std(axis=0)
        std[std == 0] = 1.0  # avoid division by zero

    ### Normalizing
    normalized_features = np.zeros_like(all_features)
    total_files = len(all_features)
    for i in range(total_files):
        normalized_features[i] = (all_features[i] - mean) / std
        if verbose and (i + 1) % 10 == 0:
            print(f"Normalized {i + 1}/{total_files} files...")

    if verbose:
        print(f"Completed normalization for {total_files} files.")

    ### Rebuilding dictionary
    normalized_dict = {key: normalized_features[i] for i, key in enumerate(keys_order)}

    return normalized_dict, mean, std

def plot_normalized_fft(norm_data_dict, fft_data_dict, key, save_path=None):

    ### Getting normalized vector
    norm_vector = norm_data_dict[key]

    ### Getting frequency axis from original FFT
    freqs = fft_data_dict[key]['freqs']
    num_freqs = len(freqs)

    ### Splitting into X, Y, Z axes
    X_norm = norm_vector[:num_freqs]
    Y_norm = norm_vector[num_freqs:2*num_freqs]
    Z_norm = norm_vector[2*num_freqs:]

    ### Creating figure
    plt.figure(figsize=(12, 10))

    #### X axis
    plt.subplot(3, 1, 1)
    plt.plot(freqs, X_norm, color='red')
    plt.title(f'Experiment {key[0]} – X axis (Normalized FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.grid(True)

    #### Y axis
    plt.subplot(3, 1, 2)
    plt.plot(freqs, Y_norm, color='green')
    plt.title(f'Experiment {key[0]} – Y axis (Normalized FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.grid(True)

    #### Z axis
    plt.subplot(3, 1, 3)
    plt.plot(freqs, Z_norm, color='blue')
    plt.title(f'Experiment {key[0]} – Z axis (Normalized FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.grid(True)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def dict_to_features_labels(norm_dict):
    keys = sorted(norm_dict.keys())
    X = np.vstack([norm_dict[k] for k in keys])
    y = np.array([k[1] for k in keys])  # k[1] is the label (0 or 1)
    return X, y, keys

def apply_pca(train_dict, valid_dict, test_dict, n_components=0.95, verbose=True):

    ### Converting dicts to feature matrices
    if verbose: print("Converting training data to feature matrix...")
    X_train, y_train, train_keys = dict_to_features_labels(train_dict)
    X_valid, y_valid, valid_keys = dict_to_features_labels(valid_dict)
    X_test,  y_test,  test_keys  = dict_to_features_labels(test_dict)

    ### Fitting PCA on training data
    if verbose: print("Fitting PCA on training data...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    if verbose:
        print(f"PCA fitted. Original dims: {X_train.shape[1]}, Reduced dims: {X_train_pca.shape[1]}")

    ### Transforming validation and test data
    X_valid_pca = pca.transform(X_valid)
    X_test_pca  = pca.transform(X_test)
    if verbose:
        print("Transformed validation and test datasets using trained PCA.")

    ### Returning
    return {
        "pca_model": pca,
        "train": (X_train_pca, y_train, train_keys),
        "valid": (X_valid_pca, y_valid, valid_keys),
        "test":  (X_test_pca,  y_test,  test_keys)
    }

def analyze_pca_variance(norm_dict, max_components=None, verbose=True, save_path=None):
    X_train, y_train, train_keys = dict_to_features_labels(norm_dict)

    # Determine maximum allowed components
    max_possible = min(X_train.shape)
    if max_components is None or max_components > max_possible:
        max_components = max_possible
        if verbose:
            print(f"⚠️ max_components adjusted to {max_components} (min(n_samples, n_features))")

    # Fit PCA with full range
    pca_full = PCA(n_components=max_components)
    pca_full.fit(X_train)

    # Compute cumulative explained variance
    cum_explained = np.cumsum(pca_full.explained_variance_ratio_) * 100

    # Plot cumulative variance curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cum_explained) + 1), cum_explained, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('Cumulative Explained Variance vs. PCA Components')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    if verbose:
        for i in [10, 20, 50, 100]:
            if i < len(cum_explained):
                print(f"Components {i:3d}: {cum_explained[i - 1]:6.2f}% variance explained")

    return pca_full, cum_explained

def print_pca_summary(pca_model, n_components=10):

    total_var = np.sum(pca_model.explained_variance_ratio_) * 100
    print(f"\n🔍 PCA Summary")
    print(f"-----------------------------")
    print(f"Total components: {pca_model.n_components_}")
    print(f"Total variance explained: {total_var:.2f}%")

    print(f"\nTop {n_components} components variance ratios:")
    for i, var in enumerate(pca_model.explained_variance_ratio_[:n_components]):
        print(f"  PC{i+1:02d}: {var*100:.2f}% variance")

## EXECUTION

### Choosing the experiment we want to look at
experiment_key = (1, 0, 1000, 100)
exp_num_str = f"{experiment_key[0]:03d}"
save_raw_path = f"experiment{exp_num_str}_raw.png"
save_fft_path = f"experiment{exp_num_str}_norm_fft.png"

### Getting all raw data
data_train_raw = read_raw_data('data_gear/train/')
data_valid_raw = read_raw_data('data_gear/valid/')
data_test_raw  = read_raw_data('data_gear/test/')

### Getting the three plots of the raw data
#plot_raw_data(data_train_raw, experiment_key, save_path=save_raw_path)

### Applying FFT to all raw data
data_train_fft = apply_fft_to_dataset(data_train_raw)
data_valid_fft = apply_fft_to_dataset(data_valid_raw)
data_test_fft  = apply_fft_to_dataset(data_test_raw)

### Normalizing all data
data_train_norm, train_mean, train_std = normalize_fft_dataset(data_train_fft)
data_valid_norm, _, _ = normalize_fft_dataset(data_valid_fft, mean=train_mean, std=train_std)
data_test_norm, _, _  = normalize_fft_dataset(data_test_fft,  mean=train_mean, std=train_std)

### Getting the three plots of the normalized data
#plot_normalized_fft(data_train_norm, data_train_fft, experiment_key, save_path=save_fft_path)

### Applying PCA

#### Running PCA and choosing k
#pca_full, cum_explained = analyze_pca_variance(data_train_norm, max_components=None, verbose=True, save_path='pca_cumulative_variance.png')
chosen_k_elbow = 25   # <-- set this from the elbow in the plotted curve
chosen_k_auto = np.argmax(cum_explained >= 95) + 1  # Automatically choose 95% threshold

pca_results_elbow = apply_pca(data_train_norm, data_valid_norm, data_test_norm, n_components=chosen_k_elbow)
pca_results_auto = apply_pca(data_train_norm, data_valid_norm, data_test_norm, n_components=chosen_k_auto)

#### Printing PCA summary
print("------------- PCA Elbow results -------------")
print_pca_summary(pca_results_elbow["pca_model"])
print("------------- PCA Auto results -------------")
print_pca_summary(pca_results_auto["pca_model"])



