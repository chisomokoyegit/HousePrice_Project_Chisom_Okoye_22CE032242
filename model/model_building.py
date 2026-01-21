import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset (place house_prices.csv in the model folder)
dataset_path = os.path.join(os.path.dirname(__file__), "house_prices.csv")
df = pd.read_csv(dataset_path)

# 2. Select features
features = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "YearBuilt"
]

X = df[features]
y = df["SalePrice"]

# 3. Handle missing values
X = X.fillna(X.median())

# 4. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# 8. Save model and scaler
output_path = os.path.join(os.path.dirname(__file__), "house_price_model.pkl")
with open(output_path, "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print("✅ House price model saved successfully")


