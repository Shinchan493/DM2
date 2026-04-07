# Binary Time-Series Classification with Extreme Class Imbalance

## 📖 Project Overview
This project tackles the classification of a highly skewed, binary, noise-removed time-series dataset (sampled at 100 Hz). In this problem domain, the primary challenge is the severe imbalance between the majority class (`0`) and the minority class (`1`).

Following strict project constraints, **no data from the minority class (`1`) was dropped or up-sampled**. Instead, a targeted **Transition-Zone Aware Under-Sampling** algorithm was developed to intelligently down-sample the majority class while preserving the critical decision boundaries where the system transitions between states.

---

## 🔬 Dataset & Pipeline Characteristics

### 1. Data Preprocessing & Sliding Window
- **Raw Data:** Multiple `.csv` files representing disparate trials, containing 9 feature columns and 1 label column.
- **Sliding Window:** Time sequence data is processed using a sliding window approach to capture temporal dynamics.
  - **Window Size:** 50 samples
  - **Overlap:** 50% (stride of 25 samples)
- **Labeling Logic:** A window is labeled as `1` if strictly **> 40%** of its constituent samples are originally labeled `1`. Otherwise, it defaults to `0`.

### 2. Feature Engineering
Since raw time-series data can be noisy and high-dimensional, the 50-length sequence for each of the 9 features is collapsed using robust statistical aggregation.
- Extracted metrics: `Mean`, `Standard Deviation`, `Minimum`, `Maximum`, `Median`.
- Result: Each sliding window is represented by a single **45-dimensional feature vector** (9 columns × 5 statistics).

### 3. Transition-Zone Under-Sampling Strategy (Targeting Imbalance)
To overcome the class imbalance without artificially altering the minority class (`1`), we designed a localized filtering approach inspired by Edited Nearest Neighbor (ENN) and Tomek Link methodologies, scaled for time-series. The instructions strictly state: *"...any of the minority class (i.e., ‘1’) data should not be removed, nor should it be up-sampled."*

Our custom Under-Sampling algorithm works as follows:
- **Chronological Tracking:** The temporal sequence of the extracted windows is preserved.
- **Transition Zone Preservation (Group A - Conserved `0`s):** Windows labeled `0` that immediately precede or succeed a `1` window (within a strict proximity size buffer) represent the "transition boundaries." These are critical for the ML model to understand *when* a shift is occurring. They contain the highest informational density. Down-sampling these would destroy the model's ability to locate the transition boundary, so **these are 100% conserved.**
- **Safe Zone Down-Sampling (Group B - Removed `0`s):** Windows labeled `0` that are temporally distant from any `1` window are deemed redundant, steady-state background noise. These are **randomly under-sampled** until our training set approaches optimal balance. 

**Statistical Impact of Custom Under-Sampling:**
By targeting redundant safe-zones exclusively, the algorithm successfully down-scaled the Imbalanced majority class from **41,632 windows** to **12,096 windows**, while perfectly preserving all **4,032** minority class windows and their adjacent borderline data.

### 4. Machine Learning Models
Four distinct architectures (encompassing shallow, deep, and ensemble algorithms) are trained and compared:
1. **Random Forest (Ensemble)**
2. **Support Vector Machine - SVM (Shallow)**
3. **XGBoost (Ensemble)**
4. **Multi-Layer Perceptron - MLP (Deep)**

*Training methodology involves **5-fold Stratified Cross-Validation** via Grid Search hyperparameter tuning.*

### 5. Custom Evaluation Metrics
As required, the evaluation module was built from scratch. It mathematically assesses the predictions using:
- **Raw Confusion Matrix**
- **True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)**
- **F1 Score**
- **Balanced Accuracy**

Models are trained on two separate pipelines: **Imbalanced Data (Baseline)** and **Transition-Zone Balanced Data**, returning results for robust comparative analysis.

---

## 📂 Project Structure

```text
📦 DM2 (Workspace)
 ┣ 📜 config.py          # Global hyperparameter settings (window size, overlap, thresholds)
 ┣ 📜 features.py        # Statistical feature extraction array mapping
 ┣ 📜 loader.py          # CSV traversal, sliding window application, and progress tracking
 ┣ 📜 sampler.py         # Custom Transition-Zone under-sampling algorithm
 ┣ 📜 models.py          # Model definition, parameter grids, and Cross-Validation loops
 ┣ 📜 metrics.py         # From-scratch mathematical evaluation functions
 ┣ 📜 main.py            # Main orchestrator pipeline (Runs loading -> balancing -> training)
 ┣ 📜 Result_Report.md   # Final formatted markdown of matrices and pipeline performance
 ┣ 📜 results_summary.json # Generated data dictionary tracing precise execution statistics
 ┗ 📜 README.md          # Project documentation (This file)
```

---

## 🚀 How To Run the Pipeline

### 1. Install Dependencies
Ensure you have Python 3.x installed along with the requisite packages. Run the following command in your terminal:
```bash
pip install numpy pandas scikit-learn xgboost tqdm
```

### 2. Verify Directory Structure
Ensure your dataset is stored correctly relative to the project root:
- `DataSet/Sample_Training/` (Used for CV training/sampling)
- `DataSet/Sample_Test/` (Strict held-out validation)

### 3. Execute
Run the orchestrator file. *Note: Training SVM and MLP models using Cross-Validation can be computationally intensive and may take a few minutes.*
```bash
python main.py
```

### 4. Review Results
Once execution is complete, the application will output a dense `results_summary.json` file. The final insights, parameter comparisons, and raw confusion matrices are transcribed into the `Result_Report.md` file.