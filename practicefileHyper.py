import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow
mlflow.set_experiment('mlExpHyperParameter')

# Define parameter grid
param_dist = {
    'max_iter': [500, 1000, 1500, 2000],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1', 'l2'],
    'random_state': [40, 50, 60, 70, 80, 90, 100],
}

# Main parent run for the best model
with mlflow.start_run() as parent_run:
    # Initialize model and search
    lg = LogisticRegression()
    rv = RandomizedSearchCV(estimator=lg, param_distributions=param_dist, cv=5, n_jobs=-1, n_iter=20)
    
    # Perform search
    rv.fit(X_train, y_train)
    
    # Log best model and results
    best_params = rv.best_params_
    best_estimator = rv.best_estimator_
    
    # Evaluate best model
    y_pred = rv.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log parent run information
    mlflow.log_params(best_params)
    mlflow.log_metric('best_accuracy', accuracy)
    mlflow.sklearn.log_model(best_estimator, "best_model")
    
    # Log all combinations as child runs
    for i in range(len(rv.cv_results_['params'])):
        with mlflow.start_run(nested=True, experiment_id=parent_run.info.experiment_id):
            # Log parameters
            params = rv.cv_results_['params'][i]
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics({
                'mean_cv_score': rv.cv_results_['mean_test_score'][i],
                'std_cv_score': rv.cv_results_['std_test_score'][i],
                'rank': rv.cv_results_['rank_test_score'][i]
            })
            
            # Optionally log temporary model
            # with mlflow.start_run(nested=True):
            #     temp_model = LogisticRegression(**params).fit(X_train, y_train)
            #     mlflow.sklearn.log_model(temp_model, "model")

    # Log artifacts and register best model
    confusion_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_mat, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    mlflow.log_artifact('confusion_matrix.png')
    
    # Register best model
    log_model_uri = f"runs:/{parent_run.info.run_id}/best_model"
    registered_model = mlflow.register_model(log_model_uri, "Best_Logistic_Regression_Model")