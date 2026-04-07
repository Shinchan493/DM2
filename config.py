import os

# Data directories based on the workspace structure
DATA_DIR = "DataSet"
TRAIN_DIR = os.path.join(DATA_DIR, "Sample_Training")
TEST_DIR = os.path.join(DATA_DIR, "Sample_Test")

# Sliding Window properties
WINDOW_SIZE = 50
OVERLAP_RATIO = 0.5
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))
LABEL_THRESHOLD = 0.40

# Transition zone under-sampling configuration
# Controls how many windows before and after a '1' are inherently conserved.
TRANSITION_ZONE_SIZE = 3 

# Under-sampling ratio: (Target Majority '0' count / Minority '1' count)
# Setting this > 1.0 (e.g. 2.0 or 3.0) prevents the massive spike in False Positives
# and preserves high F1 scores while still balancing the dataset compared to the baseline.
UNDERSAMPLE_RATIO = 3.0

# Feature and Label column configuration
# "Each file has 9 features (the first two columns are not features) and one label (the last column)."
FEATURE_COLS_START = 2
LABEL_COL = -1
