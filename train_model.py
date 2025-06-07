import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pickle

# Load your CSV
df = pd.read_csv("rent_data.csv")

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("Rent", axis=1)
y = df["Rent"]

# Save the column names (important for matching input during prediction)
columns_used = X.columns
pickle.dump(columns_used, open("columns.pkl", "wb"))

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Save the trained model and scaler
pickle.dump(model, open("svr_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
