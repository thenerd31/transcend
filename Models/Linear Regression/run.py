import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(strategy='mean')
# X = imputer.fit_transform(X)


# Load the dataset
# Replace 'business_loans_data.csv' with your dataset file name
data = pd.read_csv('business_loans_data.csv')

# Preprocess the data: One-hot encode categorical variables
data = pd.get_dummies(data, columns=['RevLineCr', 'LowDoc'])

data = data.dropna()


print(data.shape)
# Define features and target variable
X = data[['LoanNr_ChkDgt', 'Term', 'NoEmp', 'NewExist',
          'CreateJob', 'RevLineCr_Y', 'LowDoc_Y', 'NAICS']]
y = data['MIS_Status']  # Target variable: Loan status (CHGOFF or PIF)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)
