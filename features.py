import numpy as np

def extract_features(window_data):
    """
    Extract statistical features from a time-series window.
    window_data shape: (WINDOW_SIZE, NUM_FEATURES)
    Returns a flat 1D array of extracted features.
    """
    # Extract robust statistical metrics across the time dimension for the 9 features
    mean = np.mean(window_data, axis=0)
    std = np.std(window_data, axis=0)
    minimum = np.min(window_data, axis=0)
    maximum = np.max(window_data, axis=0)
    median = np.median(window_data, axis=0)
    
    # Concatenate all to form a single feature vector per window
    # 9 features * 5 stats = 45 length vector
    return np.concatenate([mean, std, minimum, maximum, median])
