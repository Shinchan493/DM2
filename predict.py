import sys
import os
import pickle
import numpy as np
from loader import extract_windows_from_file, load_data_from_directory
from metrics import calculate_metrics

def predict_on_new_data(path, model_path="model.pkl"):
    """
    Given a new CSV file or a directory of datasets and a trained model, extract features 
    and yield the predicted outcomes.
    """
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found!")
        print("Please run 'python generate_model.py' first to train and save the model.")
        return
        
    if not os.path.exists(path):
        print(f"ERROR: Target path '{path}' not found!")
        return

    print(f"Loading trained model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    print(f"Processing '{path}'...")
    if os.path.isdir(path):
        X, original_labels, file_origins = load_data_from_directory(path)
    else:
        X, original_labels = extract_windows_from_file(path)
        file_origins = np.full(len(X), path) if len(X) > 0 else []
        
    if len(X) == 0:
        print("No windows could be extracted. Check the file format and size.")
        return
        
    print(f"Making predictions on {len(X)} data windows...")
    predictions = model.predict(X)
    
    print("\n--- INFERENCE OUTCOMES ---")
    if os.path.isdir(path):
        # For a whole directory, printing each line might flood the console.
        # We can summarize the results and print the overall metrics.
        zeros = sum(predictions == 0)
        ones = sum(predictions == 1)
        print(f"Total Predictions: {len(predictions)}")
        print(f"Predicted '0's: {zeros}")
        print(f"Predicted '1's: {ones}")
        
        print("\n--- Directory Metrics Summary ---")
        metrics_dict = calculate_metrics(original_labels, predictions)
        for k, v in metrics_dict.items():
            if k == "Confusion Matrix":
                print(f"{k}:\n{np.array(v)}")
            else:
                try: # handle floats
                    print(f"{k}: {float(v):.4f}")
                except (ValueError, TypeError):
                    print(f"{k}: {v}")
    else:
        # If it's just one file, print the window-by-window predictions.
        for i, pred in enumerate(predictions):
            print(f"Window {i+1:03d} -> Predicted Label: {pred}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_new_dataset_or_directory>")
        print("Example (File): python predict.py DataSet/Sample_Test/Mole_1/some_file.csv")
        print("Example (Directory): python predict.py DataSet/Sample_Test")
    else:
        new_dataset_path = sys.argv[1]
        predict_on_new_data(new_dataset_path)
