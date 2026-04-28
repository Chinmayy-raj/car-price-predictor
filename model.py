import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("car_data.csv")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
data['Car_Age'] = 2025 - data['Year']

# Drop unnecessary columns
data = data.drop(['Car_Name', 'Year'], axis=1)

# Encoding categorical
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0, 'Diesel':1, 'CNG':2})
data['Seller_Type'] = data['Seller_Type'].map({'Dealer':0, 'Individual':1})
data['Transmission'] = data['Transmission'].map({'Manual':0, 'Automatic':1})

# -----------------------------
# TARGET TRANSFORMATION (KEY FIX)
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
# MODEL (TUNED)
# -----------------------------
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=3,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION (convert back)
# -----------------------------
y_pred_log = model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)

score = r2_score(y_actual, y_pred)

print("R2 Score:", score)

# -----------------------------
# SAVE MODEL
# -----------------------------
pickle.dump(model, open('model.pkl', 'wb'))

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("car_data.csv")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
data['Car_Age'] = 2025 - data['Year']

# Drop unnecessary columns
data = data.drop(['Car_Name', 'Year'], axis=1)

# Encoding categorical
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0, 'Diesel':1, 'CNG':2})
data['Seller_Type'] = data['Seller_Type'].map({'Dealer':0, 'Individual':1})
data['Transmission'] = data['Transmission'].map({'Manual':0, 'Automatic':1})

# -----------------------------
# TARGET TRANSFORMATION (KEY FIX)
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
# MODEL (TUNED)
# -----------------------------
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=3,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION (convert back)
# -----------------------------
y_pred_log = model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)

score = r2_score(y_actual, y_pred)

print("R2 Score:", score)

# -----------------------------
# SAVE MODEL
# -----------------------------
pickle.dump(model, open('model.pkl', 'wb'))
print("✅ Model trained and saved")