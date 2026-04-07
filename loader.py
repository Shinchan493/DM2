import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import config
from features import extract_features

def extract_windows_from_file(filepath):
    """
    Apply sliding window and labeling logic over a single CSV file.
    """
    try:
        df = pd.read_csv(filepath, header=0)
    except Exception as e:
        return np.array([]), np.array([])
        
    data = df.iloc[:, config.FEATURE_COLS_START:-1].values
    labels = df.iloc[:, config.LABEL_COL].values
    
    windows_X = []
    windows_y = []
    
    num_samples = len(data)
    for start in range(0, num_samples - config.WINDOW_SIZE + 1, config.STRIDE):
        end = start + config.WINDOW_SIZE
        window_features = data[start:end]
        window_labels = labels[start:end]
        
        # 1. Feature Extraction per window
        extracted = extract_features(window_features)
        
        # 2. Window Labeling: > 40% are '1' -> labels as '1'
        ones_count = np.sum(window_labels == 1)
        label = 1 if (ones_count / config.WINDOW_SIZE) > config.LABEL_THRESHOLD else 0
        
        windows_X.append(extracted)
        windows_y.append(label)
        
    return np.array(windows_X), np.array(windows_y)

def load_data_from_directory(directory):
    """
    Traverse directory, extract features, and apply loading bars.
    """
    X_all, y_all = [], []
    file_origins = [] # Tracks which file a window belongs to for sequence analysis
    
    # Check if 'directory' path exists
    if not os.path.exists(directory):
        print(f"Directory {directory} not found. Please verify the DataSet path.")
        return np.array([]), np.array([]), np.array([])
        
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    # Progress bar for Loading Files
    for folder in tqdm(folders, desc=f"Loading data from '{os.path.basename(directory)}' Moles", leave=True):
        folder_path = os.path.join(directory, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        for file in files:
            filepath = os.path.join(folder_path, file)
            X, y = extract_windows_from_file(filepath)
            
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
                file_origins.append(np.full(len(y), filepath))
                
    if len(X_all) == 0:
        return np.array([]), np.array([]), np.array([])
        
    return np.vstack(X_all), np.concatenate(y_all), np.concatenate(file_origins)
