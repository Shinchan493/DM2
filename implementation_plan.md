# Implementation Plan and Product Requirements Document (PRD)

## 1. Project Overview & Objectives
The goal of this project is to build, evaluate, and compare four supervised classification models on a highly skewed, noise-removed time-series dataset. The data represents a binary classification problem where the minority class ('1') is heavily outweighed by the majority class ('0'). A key requirement is to implement a strict data conservation and under-sampling strategy, evaluate results thoroughly using custom metrics, and compare the performance of imbalanced data vs. balanced data.

## 2. Core Constraints & Imbalance Strategy
Based on the project requirements and our targeted strategy, the data handling constraints are:
1. **100% Conservation of Class '1':** No minority class data will be removed or up-sampled under any circumstances.
2. **Transition-Zone Aware Under-Sampling of Class '0':** 
   - **Transition Zones:** These represent the boundaries where the physical or temporal state shifts from '0' to '1' or back. Data points (windows) belonging to the majority class ('0') that are chronologically adjacent to or near the minority class windows contain critical borderline information. These will be strictly conserved.
   - **Safe Zones (Non-Transition '0's):** Majority class windows that are temporally distant from any minority class occurrences are considered "safe" or redundant. Random under-sampling will be exclusively applied to these safe-zone windows until the desired class balance (e.g., 1:1 or 1.5:1 ratio) is achieved.

## 3. Step-by-Step Implementation Pipeline

### Phase 1: Data Preprocessing & Windowing
- **Data Loading:** Read `.csv` files from the specified folder structure. Skip the first two columns (as they are not features) and utilize the remaining 9 feature columns and 1 label column.
- **Sliding Window Application:**
  - **Window Size:** 50 samples.
  - **Overlap:** 50% (stride of 25 samples).
- **Labeling Logic:**
  - Calculate the percentage of '1's in the 50-sample window.
  - If `> 40%` of samples are '1', assign label **`1`** to the window.
  - Otherwise, assign label **`0`**.
- **Feature Extraction:** For each window, extract statistical and temporal features across the 9 feature channels (e.g., Mean, Variance, Standard Deviation, Min, Max, Skewness, Kurtosis). These features will represent the sample for the ML models.

### Phase 2: Train/Test Splitting
- Select **exactly 6 folders** that contain both classes to act as the held-out Test Set.
- The remaining folders will be aggregated to form the Training Set.

### Phase 3: Transition-Zone Under-Sampling Implementation
- **Identify Sequences:** Maintain the chronological sequence of the extracted windows within each file.
- **Locate Transition Boundaries:** Calculate the temporal distance of every '0' window to the nearest '1' window.
- **Partition Class '0':**
  - Group A (Conserved): '0' windows where `distance <= threshold` (e.g., within 2-3 windows of a '1').
  - Group B (Under-sampled): '0' windows where `distance > threshold`.
- **Apply Under-sampling:** Randomly drop samples from Group B until the sum of Group A + retained Group B achieves our target class ratio against the conserved Class '1' samples.

### Phase 4: Model Development & Tuning
Select 4 supervised classifiers to cover shallow, deep, and ensemble methods. For example:
1. Random Forest (Ensemble/Tree)
2. Support Vector Machine - SVM (Shallow/Margin-based)
3. XGBoost / Gradient Boosting (Ensemble/Boosting)
4. Multi-Layer Perceptron - MLP (Deep Learning)

- **Cross-Validation:** Use **Stratified k-fold cross-validation** (e.g., k=5) on the training set.
- **Hyperparameter Tuning:** Use Grid Search or Randomized Search to find optimal configurations for each classifier.

### Phase 5: Evaluation & Metrics (Written from Scratch)
Evaluate the models in two scenarios:
1. **Without Balancing:** Training entirely on the raw, imbalanced windowed data.
2. **With Custom Balancing:** Training on the Transition-Zone Under-sampled data.

Construct a custom evaluation module from scratch returning:
- Raw Confusion Matrix
- True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN)
- F1 Score
- Balanced Accuracy

### Phase 6: Codebase Architecture
The project will be built modularly to ensure clean logic and reusability:
- `config.py`: Global variables, window lengths, overlap ratios, data paths.
- `loader.py`: Functions to traverse directories, read CSVs, and extract sliding windows.
- `features.py`: Statistical feature extraction for time-series windows.
- `sampler.py`: Logic for computing window distances and executing the Transition-Zone Under-sampling.
- `models.py`: Classifier definitions and hyperparameter grids.
- `metrics.py`: Custom-built matrix and scoring functions.
- `main.py`: Orchestrator to run the complete pipeline and output comparative tables/charts.