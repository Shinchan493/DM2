import pickle
import time
from sklearn.model_selection import GridSearchCV
import config
from loader import load_data_from_directory
from sampler import transition_zone_under_sample
from models import get_models

def main():
    print("Loading data from train directory...")
    # If you want to train on EVERYTHING including test data, you can load TEST_DIR too.
    # For now, we follow the pipeline's balanced training approach.
    X_train_imbalanced, y_train_imbalanced, file_origins_train = load_data_from_directory(config.TRAIN_DIR)
    
    if len(X_train_imbalanced) == 0:
        print("CRITICAL: Failed to load dataset. Check file paths in config.")
        return

    print("Applying transition-zone balancing...")
    X_train_balanced, y_train_balanced = transition_zone_under_sample(
        X_train_imbalanced, 
        y_train_imbalanced, 
        file_origins_train
    )

    models = get_models()
    # Let's pick XGBoost as it consistently showed high performance in your summary
    xgb_info = models["XGBoost (Ensemble)"]
    
    print("Training XGBoost (Ensemble) on balanced data... This might take a bit.")
    clf = GridSearchCV(
        estimator=xgb_info["model"], 
        param_grid=xgb_info["params"], 
        cv=5, # 5-fold Cross-Validation
        scoring='balanced_accuracy', 
        n_jobs=-1
    )
    
    clf.fit(X_train_balanced, y_train_balanced)
    
    best_model = clf.best_estimator_
    print(f"Best parameters found: {clf.best_params_}")
    
    print("Saving the best model to model.pkl...")
    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)
        
    print("Saved successfully! You can now use predict.py to predict on new datasets.")

if __name__ == '__main__':
    main()
