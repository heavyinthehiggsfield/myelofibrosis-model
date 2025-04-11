# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Load your CSV data
df = pd.read_csv("data/external/synthetic_myelofibrosis_data_n350.csv")  # adjust path if needed

# Drop Patient ID (not useful as a feature for ML)
df.drop("Patient_ID", axis=1, inplace=True)

# Split features and target
X = df.drop("Days_to_Event", axis=1)
y = df["Days_to_Event"]

# Optional: one-hot encoding if any columns are categorical (none in current data)
# X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost regressor
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
# model = RandomForestRegressor(
#     n_estimators=100,
#     max_depth=6,
#     random_state=42
# )

model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# (Optional) Plot actual vs predicted
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Days to Event")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.show()
