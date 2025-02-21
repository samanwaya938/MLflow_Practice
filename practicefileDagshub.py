import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

X = load_iris().data
y = load_iris().target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import dagshub
dagshub.init(repo_owner='samanwaya938', repo_name='MLflow_Practice', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/samanwaya938/MLflow_Practice.mlflow')


mlflow.set_experiment('mlExp')

with mlflow.start_run():

  lg = LogisticRegression(max_iter=1000, random_state=40, solver='lbfgs', multi_class='auto',  penalty='l2', verbose=1)
  lg.fit(X_train, y_train)
  y_pred = lg.predict(X_test)
  accouracy = accuracy_score(y_test, y_pred)
  class_report = classification_report(y_test, y_pred)
  print(f'Accuracy: {accouracy}')
  print(f'Classification Report:\n{class_report}')

  confusion_mat = confusion_matrix(y_test, y_pred)
  print(f'Confusion Matrix:\n{confusion_mat}') 

  plt.figure(figsize=(10, 7))
  sns.heatmap(confusion_mat, annot=True)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.savefig('confusion_matrix.png')
  plt.show()

  

  mlflow.log_param('solver', 'ibp')
  mlflow.log_metric('accuracy', accouracy)
  mlflow.sklearn.log_model(lg, 'logistic_regression', input_example=X_test[:5])

  
  mlflow.log_artifact('confusion_matrix.png')

  log_model_uri = "runs:/{}/Logistic Regression Model".format(mlflow.active_run().info.run_id)
  registered_model_name = mlflow.register_model(log_model_uri, "Logistic Regression Model")

logisticregression_model_uri = "models:/Logistic Regression Model/latest"
loaded_lr = mlflow.sklearn.load_model(logisticregression_model_uri)