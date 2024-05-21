# Name: Dana Haham
# ID: 209278407

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

plt.ion()

# 1. 
# Familiarization.

# 1.1.
# Load train.csv
def load_train_data():
    df_train = pd.read_csv('train.csv', encoding='ISO-8859-1')
    return df_train

# 1.2.
# Display some data - display the top 10 rows
def disp_some_data(df_train):
    print(f"The top 10 rows of the dataset:\n {df_train.head(10)}")

# 1.3.
# In order to know what to do with which columns, we must know what types are there, and how many different values are there for each.
def display_column_data(df_train, max_vals=10):

    # Investigate the columns in the dataset columns
    print(df_train.info(verbose=True))

    # Count the number of unique values per column:
    num_uq_vals_sr = df_train.nunique()
    print(num_uq_vals_sr)

    # For columns that have less than max_vals values, print the number of occurrences of each value
    columns_to_print = num_uq_vals_sr[num_uq_vals_sr < max_vals].index

    for col in columns_to_print:
        print('{:s}: '.format(col), dict(df_train[col].value_counts()))

# 1.4
# Drop columns that we do not know how to handle such as free text.
def drop_non_inform_columns(df_train):

    # Store results in df_lean
    df_lean = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    return df_lean


# 2. 
# Now that we know the basics about our dataset, we can start cleaning & transforming it towards enabling prediction of survival

# 2.1
# In which columns are there missing values?
def where_are_the_nans(df_lean):

    # Print and return the names of the columns that have at least one missing value, and the number of missing values
    # Store the results in a dict or a series, where the index/key is the column name, and the value is the number of nans.
    cols_with_nans = dict()

    for col in df_lean.columns[df_lean.isnull().any()]:
        cols_with_nans[col] = df_lean[col].isnull().sum()

    print(cols_with_nans)
    return cols_with_nans

# 2.2
# Fill the missing values in the columns 'Age' and 'Embarked'.
# 'Age' with the average and 'Embarked' with the most common
def fill_titanic_nas(df_lean):

    # Copy the given dataframe
    df_filled = df_lean

    # Fill missing values in 'Age' with the mean age
    df_filled['Age'].fillna(df_lean['Age'].mean(), inplace=True)

    # Fill missing values in 'Embarked' with the most common port
    most_common_port = df_lean['Embarked'].value_counts().idxmax()
    df_filled['Embarked'].fillna(most_common_port, inplace=True)

    return df_filled

# 2.3
# Convert the non-numerical (categorical) variables
# to some numeric representation - so we can apply numerical schemes to it.
# We'll encode "Embarked" and "Pclass", using the "one-hot" method
def encode_one_hot(df_filled):

    # Apply one-hot to "Embarked" and "Pclass" columns.
    df_one_hot = pd.get_dummies(df_filled, columns=['Embarked', 'Pclass'])

    # Rename columns
    df_one_hot.rename(columns={'Embarked_C': 'Emb_C', 'Embarked_S': 'Emb_S', 'Embarked_Q': 'Emb_Q'}, inplace=True)
    df_one_hot.rename(columns={'Pclass_1': 'Cls_1', 'Pclass_2': 'Cls_2', 'Pclass_3': 'Cls_3'}, inplace=True)

    # For lack of better place, introduce a new column, "Bin_Sex" - a binary (1 or 0) version of the "Sex" column
    df_one_hot['Bin_Sex'] = df_one_hot['Sex'].map({'male': 1, 'female': 0})

    # After encoding by one-hot, we may delete the original columns
    df_one_hot = df_one_hot.drop(['Sex'], axis=1)

    return df_one_hot

# 2.4
# There are 2 variables (columns) that reflect co-travelling family of each passenger.
# SibSp - the number of sibling - brothers and sisters.
# Parch - the total number of parents plus children for each passenger.
# We want to reflect the whole family size of each passenger - the sum of SibSp and Parch
def make_family(df_one_hot):

    # Introduce a new column with the name "Family" - The sum of "SibSp" and "Parch" columns
    df_one_hot['Family'] = df_one_hot['SibSp'] + df_one_hot['Parch']

    return df_one_hot

# 2.5 Feature Transformation
# To capture that notion, we take the log of the variables of interest. To guard against taking log of 0, we add 1 prior to that.
# X -> log(1+X)
def add_log1p(df_one_hot):

    # For each of the numeric columns: 'Age', 'SibSp', 'Parch', 'Fare', 'Family'
    # We introduce a new column that starts with the 'log1p_' string: 'log1p_Age', 'log1p_SibSp', 'log1p_Parch', 'log1p_Fare', 'log1p_Family'
    for col in ['Age', 'SibSp', 'Parch', 'Fare', 'Family']:
        df_one_hot['log1p_' + col] = np.log1p(df_one_hot[col] + 1)

    return df_one_hot


# 3.
# Basic exploration of survival. 
# This section deals with correlations of the "Survived" column to various other data about the passengers.

# 3.1. Survival vs gender
def survival_vs_gender(df):

    # What is the survival rate for women and men?

    # Filter DataFrame for male and female passengers
    df_male = df[df['Bin_Sex'] == 1]
    df_female = df[df['Bin_Sex'] == 0]

    # Compute survival rate for men and women
    survival_rate_male = df_male['Survived'].mean()
    survival_rate_female = df_female['Survived'].mean()

    survived_by_gender = {'male': survival_rate_male, 'female': survival_rate_female}

    print(survived_by_gender)
    return survived_by_gender

# 3.2 The same for survival by class using the "one-hot" encoding
def survival_vs_class(df):

    # Filter DataFrame according to the class of the passengers
    df_cls1 = df[df['Cls_1'] == 1]
    df_cls2 = df[df['Cls_2'] == 1]
    df_cls3 = df[df['Cls_3'] == 1]

    # Compute survival rate according to the class of the passengers
    survival_rate_cls1 = df_cls1['Survived'].mean()
    survival_rate_cls2 = df_cls2['Survived'].mean()
    survival_rate_cls3 = df_cls3['Survived'].mean()

    survived_by_class = {'Cls_1': survival_rate_cls1, 'Cls_2': survival_rate_cls2, "Cls_3": survival_rate_cls3}
    print(survived_by_class)

    return survived_by_class

# 3.3 The same, for survival by the three family size metrics. Return a dict of dicts / series
def survival_vs_family(df):
    
    # The different family size metrics - "SibSp", "Parch", "Family" are all numeric.
    survived_by_family = {}

    for metric in ['SibSp', 'Parch', 'Family']:

        survived_by_metric = {}

        for value in df[metric].unique():

            # Filter DataFrame for every unique value in the metric colum
            df_value = df[df[metric] == value]

            # Compute survival rate
            survival_rate_value = df_value['Survived'].mean()

            survived_by_metric[value] = survival_rate_value

        print("Family metric: ", metric)
        print("Survival stats:")
        print(survived_by_metric)
        
        survived_by_family[metric] = survived_by_metric

        max_value = max(survived_by_metric, key=lambda k: survived_by_metric[k])

      # What survival metric with what value ensures the highest probability of survival?
        print("To ensure the highest chance of survival, the metric ", metric,
              'must have the value ', max_value)
       
    return survived_by_family

# 3.4 Visualizing the distribution of age and its impact on survival
def survival_vs_age(df):

    # Here we would like to plot some histograms.
    bins = list(range(0, 100, 4))

    plt.close('Age, all')
    plt.figure('Age, all')
    df['Age'].hist(bins=bins)
    plt.title('Age Distribution for All Passengers')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()

    # For Survivors
    plt.close('Age, Survivors')
    plt.figure('Age, Survivors')
    df[df['Survived'] == 1]['Age'].hist(bins=bins, alpha=0.5, label='Survived')
    plt.title('Age Distribution of Survivors')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()

    # For Non-Survivors
    plt.close('Age, Non-Survivors')
    plt.figure('Age, Non-Survivors')
    df[df['Survived'] == 0]['Age'].hist(bins=bins, alpha=0.5, color='red', label='Not Survived')
    plt.title('Age Distribution of Non-Survivors')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()

    # Bonus 1: Plot histograms based on gender
    plt.close('Age, Survived Male')
    plt.figure('Age, Survived Male')
    df[(df['Bin_Sex'] == 1) & (df['Survived'] == 1)]['Age'].hist(bins=bins, alpha=0.5, label='Male')
    plt.title('Age Distribution by Survived Male')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()

    plt.close('Age, Non-Survived Male')
    plt.figure('Age, Non-Survived Male')
    df[(df['Bin_Sex'] == 1) & (df['Survived'] == 0)]['Age'].hist(bins=bins, alpha=0.5, label='Male')
    plt.title('Age Distribution by Non-Survived Male')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()

    plt.close('Age, Survived Female')
    plt.figure('Age, Survived Female')
    df[(df['Bin_Sex'] == 0) & (df['Survived'] == 1)]['Age'].hist(bins=bins, alpha=0.5, label='Female')
    plt.title('Age Distribution by Survived Female')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()

    plt.close('Age, Non-Survived Female')
    plt.figure('Age, Non-Survived Female')
    df[(df['Bin_Sex'] == 0) & (df['Survived'] == 0)]['Age'].hist(bins=bins, alpha=0.5, label='Female')
    plt.title('Age Distribution by Non-Survived Female')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()

    # Bonus 2: Plot histograms based on passenger class
    for i in range(3):
        plt.close(f'Age, Survived Passenger Class {i+1}')
        plt.figure(f'Age, Survived Passenger Class {i+1}')
        df[(df[f'Cls_{i+1}'] == 1) & (df['Survived'] == 1)]['Age'].hist(bins=bins, alpha=0.5, label=f'Class {i+1}')
        plt.title(f'Age Distribution by Survived Passenger Class {i+1}')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.legend()

    for i in range(3):
        plt.close(f'Age, Non-Survived Passenger Class {i+1}')
        plt.figure(f'Age, Non-Survived Passenger Class {i+1}')
        df[(df[f'Cls_{i+1}'] == 1) & (df['Survived'] == 0)]['Age'].hist(bins=bins, alpha=0.5, label=f'Class {i+1}')
        plt.title(f'Age Distribution by None-Survived Passenger Class {i+1}')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.legend()

    plt.show()

# 3.5 Correlation of survival to the numerical variables
# ['Age', 'SibSp', 'Parch', 'Fare', 'Family']
# ['log1p_Age', 'log1p_SibSp', 'log1p_Parch', 'log1p_Fare', 'log1p_Family']
def survival_correlations(df, n):

    # We can compute the correlation of the various numeric columns to survival.
    corr = df.corr()

    # corr is a DataFrame that represents the correlation matrix
    print(corr)

    # Extract the correlation of each feature to the "Survived" column
    corr_to_survival = corr['Survived'].drop(['Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Family'])

    # Sort the absolute correlation values in descending order
    sorted_corr = corr_to_survival.abs().sort_values(ascending=False)
    
    # Select the n most important numerical columns
    important_feats = sorted_corr[:n]

    # Store the correlations of the selected columns
    important_corrs = {feat: corr_to_survival[feat] for feat in important_feats.index}
    
    print(important_corrs)
    return important_corrs


# 4.
# Predicting survival!!!
# Build a model and predict survival! 


# 4.1 split data into train and test sets
def split_data(df_one_hot):

    Y = df_one_hot['Survived']
    X = df_one_hot.drop(['Survived'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y)

    # This splits the data into a train set, which will be used to calibrate the internal parameters of predictor, and the test set, which will be used for checking
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test

# 4.2 Training and testing
def train_logistic_regression(X_train, X_test, y_train, y_test):

    para_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 50], # internal regularization parameter of LogisticRegression
                 'solver': ['sag', 'saga']}

    Logit1 = GridSearchCV(LogisticRegression(penalty='l2', random_state=1), para_grid, cv=5)

    Logit1.fit(X_train, y_train)

    y_test_logistic = Logit1.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_test_logistic)
    accuracy = accuracy_score(y_test, y_test_logistic)
    f1 = f1_score(y_test, y_test_logistic)

    print('acc: ', accuracy, 'f1: ', f1)
    print('confusion matrix:\n', conf_matrix)
    
    return accuracy, f1, conf_matrix


if __name__ == '__main__':
  
    df_train = load_train_data()
    disp_some_data(df_train)
    display_column_data(df_train, max_vals=10)
    df_lean = drop_non_inform_columns(df_train)
    
    cols_with_nans = where_are_the_nans(df_lean)
    df_filled = fill_titanic_nas(df_lean)
    df_one_hot = encode_one_hot(df_filled)
    df_one_hot = make_family(df_one_hot)
    df_one_hot = add_log1p(df_one_hot)
 
    survived_by_gender = survival_vs_gender(df_one_hot)
    survived_by_class = survival_vs_class(df_one_hot)
    survived_by_family = survival_vs_family(df_one_hot)
    survival_vs_age(df_one_hot)

    important_corrs = survival_correlations(df_one_hot, n=4)
    important_cols = list(important_corrs.keys())
    important_cols.append('Survived')
    df_one_hot = df_one_hot[important_cols]
    
    X_train, X_test, y_train, y_test = split_data(df_one_hot)
    train_logistic_regression(X_train, X_test, y_train, y_test)

    






