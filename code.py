# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 22:16:02 2024

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

class FraudDetectionModel:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.numeric_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        self.categorical_features = ['type', 'isFlaggedFraud']

    def data_summary(self):
        print("Shape of dataset:", self.df.shape)
        print("Head of dataset:\n", self.df.head())
        print("Missing values:\n", self.df.isnull().sum())
        print("Data types:\n", self.df.dtypes)
        print("Summary Statistics:\n", self.df.describe())

    def visualize_data(self):
        # Visualizing Numerical features with Histogram plots
        plt.figure(figsize=(12, 8))
        self.df[self.numeric_features].hist(bins=50)
        plt.tight_layout()
        plt.show()

        # Visualizing Numerical features with Box plots
        plt.figure(figsize=(12, 8))
        self.df[self.numeric_features].plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
        plt.tight_layout()
        plt.show()

    def preprocess_data(self, X):
        # Apply Winsorization to each numeric feature
        for feature in self.numeric_features:
            X[feature] = winsorize(X[feature], limits=[0.10, 0.10])
        
        # Apply One-Hot Encoding to the categorical features
        X = pd.get_dummies(X, columns=self.categorical_features)
        return X

    def split_data(self):
        # Drop high cardinality features
        self.df = self.df.drop(columns=['nameOrig', 'nameDest'])

        # Separate features and target variable
        X = self.df.drop(columns=['isFraud'])
        y = self.df['isFraud']

        # First split: 60% training and 40% temporary
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

        # Second split: 50% validation and 50% test from the temporary set (results in 20% each of original data)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_logistic_regression(self, X_train, y_train):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return model

    def train_decision_tree(self, X_train, y_train):
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train, y_train):
        model = XGBClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    def visualize_correlation(self, X):
        corr_matrix = X.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='Reds')
        plt.title('Correlation Heatmap')
        plt.show()

import unittest

class TestFraudDetectionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FraudDetectionModel("path_to_your_file/Fraud.csv")

    def test_data_summary(self):
        self.model.data_summary()
        self.assertEqual(self.model.df.shape[0], 6362620)  # Replace with actual shape for your data
        self.assertEqual(self.model.df.isnull().sum().sum(), 0)  # No missing values

    def test_preprocess_data(self):
        X_train, _, _, _, _, _ = self.model.split_data()
        X_train_preprocessed = self.model.preprocess_data(X_train.copy())
        self.assertIn('amount', X_train_preprocessed.columns)
        self.assertIn('type_CASH_IN', X_train_preprocessed.columns)

    def test_logistic_regression(self):
        X_train, X_val, _, y_train, y_val, _ = self.model.split_data()
        X_train_preprocessed = self.model.preprocess_data(X_train.copy())
        X_val_preprocessed = self.model.preprocess_data(X_val.copy())
        model = self.model.train_logistic_regression(X_train_preprocessed, y_train)
        self.model.evaluate_model(model, X_val_preprocessed, y_val)

    def test_decision_tree(self):
        X_train, X_val, _, y_train, y_val, _ = self.model.split_data()
        X_train_preprocessed = self.model.preprocess_data(X_train.copy())
        X_val_preprocessed = self.model.preprocess_data(X_val.copy())
        model = self.model.train_decision_tree(X_train_preprocessed, y_train)
        self.model.evaluate_model(model, X_val_preprocessed, y_val)

    def test_xgboost(self):
        X_train, X_val, _, y_train, y_val, _ = self.model.split_data()
        X_train_preprocessed = self.model.preprocess_data(X_train.copy())
        X_val_preprocessed = self.model.preprocess_data(X_val.copy())
        model = self.model.train_xgboost(X_train_preprocessed, y_train)
        self.model.evaluate_model(model, X_val_preprocessed, y_val)

if __name__ == '__main__':
    unittest.main()
