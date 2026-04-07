from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tqdm import tqdm
from metrics import calculate_metrics

def get_models():
    """ Define 4 separate classifiers combining shallow and ensemble architectures. """
    return {
        "RandomForest (Ensemble)": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [50, 100], 
                'max_depth': [None, 10, 20]
            }
        },
        "SVM (Shallow)": {
            "model": SVC(probability=False, random_state=42), # probability False for speed
            "params": {
                'C': [0.1, 1, 10], 
                'kernel': ['rbf']
            }
        },
        "XGBoost (Ensemble)": {
            "model": XGBClassifier(random_state=42, eval_metric='logloss'),
            "params": {
                'n_estimators': [50, 100], 
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7]
            }
        },
        "MLP (Deep)": {
            "model": MLPClassifier(random_state=42, max_iter=300),
            "params": {
                'hidden_layer_sizes': [(50,), (100,)], 
                'alpha': [0.0001, 0.001],
                'learning_rate_init': [0.001, 0.01]
            }
        }
    }

def train_and_evaluate(X_train, y_train, X_test, y_test, descriptor=""):
    """
    Train via Stratified k-fold and tune hyper-parameters. Evaluates metrics from scratch.
    """
    models = get_models()
    results = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"\n--- Initiating Cross-Validation Training: {descriptor} ---")
    # Progress bar wrapping the model loop for Training visibility
    for name, m_info in tqdm(models.items(), desc=f"Training {len(models)} CLF Pipelines"):
        # Grid Search implicitly runs Stratified k-fold (cv=cv below)
        clf = GridSearchCV(
            estimator=m_info["model"], 
            param_grid=m_info["params"], 
            cv=cv, 
            scoring='balanced_accuracy', 
            n_jobs=-1
        )
        
        # Executes Hyper-parameter tuning and Cross-Validation
        clf.fit(X_train, y_train)
        
        # Refit on strongest parameters
        best_model = clf.best_estimator_
        
        # Test evaluation strictly returned using our own module
        y_pred = best_model.predict(X_test)
        metrics_dict = calculate_metrics(y_test, y_pred)
        
        results[name] = {
            "best_params": clf.best_params_, 
            "metrics": metrics_dict
        }
        
    return results
