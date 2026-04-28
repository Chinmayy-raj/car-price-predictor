import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("car_data.csv")

# Clean column names
data.columns = data.columns.str.strip()

print("Columns in dataset:", data.columns)

# -----------------------------
# FIX NUMERIC COLUMNS (CRITICAL)
# -----------------------------
numeric_cols = ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Owner']

for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Remove invalid rows
data = data.dropna(subset=numeric_cols)

# Convert to correct types
data['Year'] = data['Year'].astype(int)
data['Owner'] = data['Owner'].astype(int)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
data['Car_Age'] = 2025 - data['Year']

# Drop unnecessary columns safely
drop_cols = [col for col in ['Car_Name', 'Year'] if col in data.columns]
data = data.drop(drop_cols, axis=1)

# -----------------------------
# ENCODING
# -----------------------------
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0, 'Diesel':1, 'CNG':2})
data['Seller_Type'] = data['Seller_Type'].map({'Dealer':0, 'Individual':1})
data['Transmission'] = data['Transmission'].map({'Manual':0, 'Automatic':1})

# -----------------------------
# TARGET TRANSFORMATION
# -----------------------------
data['Selling_Price_Log'] = np.log1p(data['Selling_Price'])

# Features & target
X = data.drop(['Selling_Price', 'Selling_Price_Log'], axis=1)
y = data['Selling_Price_Log']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=3,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# TEST EVALUATION
# -----------------------------
y_pred_log = model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)

test_score = r2_score(y_actual, y_pred)

print("Test R2 Score:", test_score)

# -----------------------------
# TRAIN & TEST R2
# -----------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)

y_train_pred_actual = np.expm1(y_train_pred)
y_test_pred_actual = np.expm1(y_test_pred)

train_r2 = r2_score(y_train_actual, y_train_pred_actual)
test_r2 = r2_score(y_test_actual, y_test_pred_actual)

print("Train R2 Score:", train_r2)
print("Test R2 Score:", test_r2)

# -----------------------------
# MODEL STABILITY CHECK
# -----------------------------
print("Difference (Overfitting Check):", abs(train_r2 - test_r2))

# -----------------------------
# SAVE MODEL
# -----------------------------
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Model trained and saved successfully")