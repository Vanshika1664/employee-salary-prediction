import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("Salary Data.csv")
df.dropna(inplace=True)

# Label encode Gender and Education Level (ordinal)
label_encoders = {}
for col in ['Gender', 'Education Level']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# One-hot encode Job Title
df = pd.get_dummies(df, columns=['Job Title'])

# Define features and target
X = df.drop(columns=['Salary'])
y = df['Salary']

# Save column order
feature_columns = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save files
joblib.dump(model, "salary_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("✅ Model trained and saved with one-hot encoding.")
