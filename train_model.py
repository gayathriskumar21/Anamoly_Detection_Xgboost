import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load datasets
train_data = pd.read_csv("Dataset/ML-MATT-CompetitionQT1920_train.csv", encoding="latin-1")

# Clean numeric columns
exclude_cols = ['Time', 'CellName']
numeric_cols = [c for c in train_data.columns if c not in exclude_cols]

for col in numeric_cols:
    train_data[col] = pd.to_numeric(train_data[col].replace('#¡VALOR!', np.nan), errors='coerce')
    median_val = train_data[col].median()
    train_data[col] = train_data[col].fillna(median_val)

# Encode 'CellName'
le = LabelEncoder()
train_data['CellName_encoded'] = le.fit_transform(train_data['CellName'])

# Extract time features
train_data['Time'] = pd.to_datetime(train_data['Time'], format='%H:%M')
train_data['Hour'] = train_data['Time'].dt.hour
train_data['DayOfWeek'] = train_data['Time'].dt.dayofweek

# Define features and target
numerical_columns = ['PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', 'maxThr_DL', 'maxThr_UL',
                     'meanUE_DL', 'meanUE_UL', 'maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL']
features = numerical_columns + ['CellName_encoded', 'Hour', 'DayOfWeek']
X_train = train_data[features]
y_train = train_data['Unusual']

# Train XGBoost with hyperparameter tuning
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Save model, label encoder, and feature names
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/xgboost_model.joblib')
joblib.dump(le, 'models/cellname_encoder.joblib')
joblib.dump(features, 'models/feature_names.joblib')

print("Model, encoder, features saved successfully.")

# Optional: evaluate on train set
train_preds = best_model.predict(X_train)
print("\nTraining set performance:")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(f"Accuracy: {accuracy_score(y_train, train_preds):.4f}")
print(f"Precision: {precision_score(y_train, train_preds):.4f}")
print(f"Recall: {recall_score(y_train, train_preds):.4f}")
print(f"F1-score: {f1_score(y_train, train_preds):.4f}")
