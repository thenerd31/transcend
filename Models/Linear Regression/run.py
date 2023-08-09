
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import numpy as np
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# imputer = SimpleImputer(strategy='mean')
# X = imputer.fit_transform(X)


# Load the dataset
# Replace 'business_loans_data.csv' with your dataset file name
data = pd.read_csv('SBAnational.csv')

# Preprocess the data: One-hot encode categorical variables

def preprocessing():
    # Removes the dollar signs and commas
    def custom_dollar_converter(dollar_str):
        if '#' in dollar_str:
            return np.nan
        else:
            dollar_str = dollar_str.replace('$', '').replace(',', '')
            return float(dollar_str)

    # Determines whether a business is a franchise or not
    # If the value is 0 or 1, the business is not a franchise
    def custom_franchise_converter(franchise_str):
        if franchise_str.strip() == '0' or franchise_str.strip() == '1':
            return 0
        else:
            return 1

    # Trims zip codes to the first two numbers
    def zip_trimmer(zip_str):
        return zip_str[:2]

    # reads in data, using the custom converters
    bank_converters = {
        'DisbursementGross': custom_dollar_converter,
        'BalanceGross': custom_dollar_converter,
        'ChgOffPrinGr': custom_dollar_converter,
        'GrAppv': custom_dollar_converter,
        'SBA_Appv': custom_dollar_converter,
        'FranchiseCode': custom_franchise_converter,
        'Zip': zip_trimmer,
    }

    bank_parse_dates = ['ApprovalDate', 'ChgOffDate', 'DisbursementDate']

    df = pd.read_csv(
        'SBAnational.csv',
        converters=bank_converters,
        parse_dates=bank_parse_dates,
        date_parser=pd.to_datetime,
    )

    # drops unnecessary columns
    drop_columns = [
        'Name', 'City', 'ChgOffDate', 'DisbursementDate',
        'LoanNr_ChkDgt', 'Bank', 'NAICS',
        'CreateJob', 'RetainedJob', 'ChgOffPrinGr',
        'RevLineCr', 'LowDoc',
    ]
    working_df = df.drop(columns=drop_columns)

    # removes all null values
    working_df = working_df.dropna()

    # label encode MIS_Status
    mis_label_encoder = LabelEncoder()
    mis_encoded = mis_label_encoder.fit_transform(working_df['MIS_Status'])
    working_df['MIS_Status'] = mis_encoded

    # makes the NewExist variable more intuitive
    # a value of 1 means the business is new
    # a value of 0 means the business is not new
    working_df['NewExist'] = working_df['NewExist'].replace({2: 1, 1: 0})
    # one hot encoding NewExist
    new_exist_true = working_df['NewExist'] == 1
    new_exist_false = working_df['NewExist'] == 0
    working_df['NewExistTrue'] = new_exist_true
    working_df['NewExistFalse'] = new_exist_false
    working_df = working_df.drop(columns=['NewExist'])

    # handling datetime information
    approval_date_months = working_df['ApprovalDate'].dt.month
    approval_date_days = working_df['ApprovalDate'].dt.day
    working_df['ApprovalMonth'] = approval_date_months
    working_df['ApprovalDay'] = approval_date_days
    working_df = working_df.drop(columns=['ApprovalDate'])

    approval_years = []
    for date in working_df['ApprovalFY']:
        if date == '1976A':
            approval_years.append(1976)
        else:
            approval_years.append(int(date))

    working_df['ApprovalFY'] = np.array(approval_years).astype(np.int64)

    # label encoding state information
    state_label_encoder = LabelEncoder()
    state_encoded = state_label_encoder.fit_transform(working_df['State'])
    bank_state_encoded = state_label_encoder.fit_transform(
        working_df['BankState'])
    working_df['State'] = state_encoded
    working_df['BankState'] = bank_state_encoded

    # converting zip information to the right datatype
    working_df['Zip'] = pd.to_numeric(working_df['Zip'])

    # separate data by features and target
    X = working_df.drop(columns=['MIS_Status'])
    y = working_df['MIS_Status']
    # separate the testing data from the training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # balancing the training data based on MIS_Status
    sampler = RandomOverSampler(sampling_strategy='minority')
    df_without_status, df_with_status = X_train, y_train
    df_without_status_rebalanced, df_with_status_rebalanced = sampler.fit_resample(
        df_without_status, df_with_status
    )
    X_train, y_train = df_without_status_rebalanced, df_with_status_rebalanced

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = preprocessing()

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
