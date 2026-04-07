import config
from loader import load_data_from_directory
from sampler import transition_zone_under_sample
from models import train_and_evaluate
import json
import time

def print_separator():
    print("-" * 60)

def main():
    print_separator()
    print("[1] INITIATING SLIDING WINDOW & PREPROCESSING...")
    print_separator()
    
    # 1. Load Data
    X_train_imbalanced, y_train_imbalanced, file_origins_train = load_data_from_directory(config.TRAIN_DIR)
    X_test, y_test, file_origins_test = load_data_from_directory(config.TEST_DIR)

    if len(X_train_imbalanced) == 0 or len(X_test) == 0:
        print("CRITICAL: Failed to load dataset. Check file paths in setup.")
        return

    # 2. Applying Transition-Zone Data Balancing (Under-Sampling)
    print_separator()
    print("[2] APPLYING TRANSITION-ZONE BALANCING ALGORITHM (UNDER-SAMPLING)...")
    print_separator()
    
    start_time = time.time()
    X_train_balanced, y_train_balanced = transition_zone_under_sample(
        X_train_imbalanced, 
        y_train_imbalanced, 
        file_origins_train
    )
    
    # Validation logging
    print(f">> Raw Imbalanced Train:  Shape: {X_train_imbalanced.shape} | Majority '0s': {sum(y_train_imbalanced==0)} | Minority '1s': {sum(y_train_imbalanced==1)}")
    print(f">> Final Balanced Train:  Shape: {X_train_balanced.shape} | Majority '0s': {sum(y_train_balanced==0)}   | Minority '1s': {sum(y_train_balanced==1)}")
    print(f"Time Taken for Sampler: {time.time() - start_time:.2f}s\n")
    print(f">> Testing Data Cohort:   Shape: {X_test.shape} | Majority '0s': {sum(y_test==0)}   | Minority '1s': {sum(y_test==1)}\n")
    

    # 3. Model Training & Validation Check (Stratified k-fold via GridSearchCV implied)
    print_separator()
    print("[3] ENTERING TRAINING STAGE VIA STRATIFIED K-FOLD CV (Tuning Parameters)...")
    print_separator()
    
    print("\n--- STAGE A: Imbalanced Training Path ---")
    results_imbalanced = train_and_evaluate(
        X_train_imbalanced, y_train_imbalanced, 
        X_test, y_test, 
        descriptor="RAW IMBALANCED DATA"
    )

    print("\n--- STAGE B: Transition Balanced Training Path ---")
    results_balanced = train_and_evaluate(
        X_train_balanced, y_train_balanced, 
        X_test, y_test, 
        descriptor="TRANSITION-BALANCED DATA"
    )
    
    # 4. Extracting final metrics dictionary to JSON
    training_stats = {
        "Imbalanced_Train_Size": len(X_train_imbalanced),
        "Imbalanced_Train_Majority_0": int(sum(y_train_imbalanced==0)),
        "Imbalanced_Train_Minority_1": int(sum(y_train_imbalanced==1)),
        "Balanced_Train_Size": len(X_train_balanced),
        "Balanced_Train_Majority_0": int(sum(y_train_balanced==0)),
        "Balanced_Train_Minority_1": int(sum(y_train_balanced==1)),
        "Test_Size": len(X_test),
        "Test_Majority_0": int(sum(y_test==0)),
        "Test_Minority_1": int(sum(y_test==1))
    }
    
    final_output = {
        "Data_Statistics": training_stats,
        "Baseline_Imbalanced": results_imbalanced,
        "Transition_Balanced": results_balanced
    }
    
    with open("results_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
        
    print_separator()
    print("[4] PIPELINE COMPLETED SUCCESSFULLY.")
    print("Metrics arrays, parameters, and Raw Confusion Matrix mapped to results_summary.json")
    print("Please transpose outcomes to Result_Report.md! 🎉")

if __name__ == '__main__':
    main()
