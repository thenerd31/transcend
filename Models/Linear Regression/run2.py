import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv("SBAnational.csv")  # Replace with the actual file path

# Preprocessing
data = data.dropna()  # Remove rows with missing values
data["Bias"] = (data["MIS_Status"] == "CHGOFF").astype(
    int)  # Create a binary target variable for bias

# Select features and target variable
features = ["ApprovalFY", "Term", "NoEmp", "NewExist", "CreateJob", "RetainedJob", "FranchiseCode",
            "UrbanRural", "RevLineCr", "LowDoc", "GrAppv", "SBA_Appv"]
target = "Bias"

X = data[features]
y = data[target]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Classification report
report = classification_report(y_test, y_pred)
print(report)
