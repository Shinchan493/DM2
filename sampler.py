import numpy as np
import config

def transition_zone_under_sample(X, y, file_origins):
    """
    Algorithm to conserve strictly 100% of '1's.
    Conserve transition zone '0's (near '1's chronologically).
    Downsample distant '0's randomly to match Class '1' distributions.
    """
    unique_files = np.unique(file_origins)
    
    # Create mask mapping True for windows inside the "transition zone"
    transition_mask = np.zeros(len(y), dtype=bool)
    
    for f in unique_files:
        f_indices = np.where(file_origins == f)[0]
        f_y = y[f_indices]
        
        # Locate Class 1 occurrence positions in this specific file 
        ones_positions = np.where(f_y == 1)[0]
        
        for pos in ones_positions:
            # Mark adjacent windows as True
            start_zone = max(0, pos - config.TRANSITION_ZONE_SIZE)
            end_zone = min(len(f_y), pos + config.TRANSITION_ZONE_SIZE + 1)
            transition_mask[f_indices[start_zone:end_zone]] = True
            
    # Subdivide arrays based on transition mask constraints
    group_A_indices = np.where((y == 0) & transition_mask)[0] # Conserved 0s (Transition Zone)
    group_B_indices = np.where((y == 0) & ~transition_mask)[0] # Undersampled 0s (Safe Zone)
    class_1_indices = np.where(y == 1)[0] # Conserved 1s (100%)
    
    # We want to roughly match the number of '1's to prevent models predicting '0' constantly.
    # By using UNDERSAMPLE_RATIO from config (e.g. 3.0), we preserve a mild imbalance.
    # This prevents the extreme False Positive spike and drastically raises the balanced F1 score.
    target_0s = max(int(len(class_1_indices) * config.UNDERSAMPLE_RATIO), len(group_A_indices))
    current_0s = len(group_A_indices)
    
    sampled_B_indices = []
    
    # Draw linearly from Group B to resolve deficit towards balance
    if current_0s < target_0s and len(group_B_indices) > 0:
        needed = target_0s - current_0s
        needed = min(needed, len(group_B_indices))
        # Important constraint check: Random Under-Sampling
        sampled_B_indices = np.random.choice(group_B_indices, size=needed, replace=False)
        
    final_indices = np.concatenate([class_1_indices, group_A_indices, sampled_B_indices])
    
    # Keep chronological tracking intact to potentially assist features models implicitly
    final_indices = np.sort(np.asarray(final_indices).astype(int))
    
    return X[final_indices], y[final_indices]
