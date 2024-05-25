# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:42:48 2024

@author: Admin
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis , skew
from matplotlib import pyplot as plt

data = pd.read_csv(r'F:\step\Fraud.csv')
data.info()
data.head(10)
data.columns
## check for duplicates###
## check for missing values##
data1 = data.copy()
data1 = data1.replace('',np.nan)
data1.isna().sum()
data1.isnull().sum()

##univariate statistics for each variable###
data1.shape
row_count = data1.shape[0]
column_count = data1.shape[1]
print(f"The row count of the dataset is {row_count}")
print(f"The row count of the dataset is {column_count}")
pd.set_option('display.precision',2)
pd.set_option('display.max_column',15)
data1.describe()
# check the denomination for amount
##print the unique values of each column
print(data1.step.unique())
print(data1.type.unique())
print(data1.type.value_counts())
print(data1.amount.unique())
print(data1.nameOrig.unique())
data1['nameorg_check'] = data1['nameOrig'].astype(str).str[0]
print(data1.nameorg_check.unique())
## we observe that all the name starts with letter C which means all the account belongs to a customer##
print(data1.oldbalanceOrg.unique())
print(data1.newbalanceOrig.unique())
print(data1.newbalanceOrig.value_counts())
print(data1.nameDest.unique())
data1['namedest_check'] = data1['nameDest'].astype(str).str[0]
#data1['namedest_check'] = data1['namedest_check'].astype("category")
print(data1.namedest_check.unique())
print(data1['namedest_check'].value_counts())
## we observe that all the name starts with letter C and M which means the receiver can be a customer or merchant ##
print(data1.oldbalanceDest.unique())
print(data1.newbalanceDest.unique())
print(data1.isFraud.unique())
print(data1.isFlaggedFraud.unique())

## Total categorical variables is three###
##checking for skewness
print(skew(data1.isFraud,bias=False))

boxplot = data1.boxplot(column= ['amount'])
plt.hist(data1['amount'], bins=3, color='skyblue', edgecolor='black')

##1.step##
print(data1['step'].count())
print(data1['step'].mean()) 
print(data1['step'].median())
print(data1['step'].mode())
print(data1['step'].std())
#print(data1['step'].value_counts())
print(data1['step'].max())
print(data['step'].min())
data2 = data1[data1['isFraud']==1]
data1['days'] = data1['step']//24
plt.set_xlabel('step') 
plt.hist(data1['step'], bins=20, color='skyblue', edgecolor='black')
#plt.hist(data1['days'], bins=15, color='skyblue', edgecolor='black')
plt.hist(data2['days'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel("Hours")
plt.ylabel("TransactionAmount")




#plt.bar_label(container, fontsize=20, color='navy')
plt.title = ('Step_variable Distrubtion') 
plt.show()
##2.type##
print(data1['type'].count())
#print(data1['step'].mean()) 
#print(data1['step'].median())
#print(data1['step'].mode())
#print(data1['step'].std())
print(data1['type'].value_counts())
print(data['type'].unique())
#print(data1['step'].max())
#print(data['step'].min())
plt.hist(data1['type'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel("Transaction Type")
plt.ylabel("TransactionAmount")

plt.hist(data2['type'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel("Transaction Type for target variable")

##3.amount##
print(data1['amount'].count())
print(data1['amount'].mean()) 
print(data1['amount'].median())
#print(data1['amount'].mode())
print(data1['amount'].std())
#print(data1['amount'].value_counts())
#print(data['amount'].unique())
print(data1['amount'].max())

binse1 = np.arange(0, max(data1.amount)+4e4, 4e4)
plt.figure(figsize=(14,6))
plt.hist(data = data1.sample(10000), x='amount', bins=binse1, kde=True)
plt.title('Distribution of the Transaction amount')
plt.xlabel('Amount sent by the transaction, n')
plt.xlim(0,2e7);
plt.ylim(0,2e7);



#print(data['step'].min())
plt.hist(data1['amount'], bins=20, color='skyblue', edgecolor='black')
plt.xlim(0,2e5)
plt.ylim(0,2e7)
plt.hist(data2['amount'], bins=15, color='skyblue', edgecolor='black')

##checking for outlier##
boxplot = data1.boxplot(column= ['amount'])
# IQR
Q1 = np.percentile(data1['amount'], 25, method='midpoint')
Q3 = np.percentile(data1['amount'], 75, method='midpoint')
IQR = Q3 - Q1
print(IQR)
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

upper_array = np.where(data1['amount'] >= upper)[0]
lower_array = np.where(data1['amount'] <= lower)[0]
 
# Removing the outliers
data3 = data1['amount'].drop(index=upper_array, inplace=True)

#outliers = data1[((data1['amount']<(Q1-1.5*IQR)) | (data1['amount']>(Q3+1.5*IQR)))]
not_outliers = data1[~((data1['amount']<(Q1-1.5*IQR)) | (data1['amount']>(Q3+1.5*IQR)))]
not_outliers.shape
##4.nameOrig## 
print(data1.nameOrig.unique().sum())
data1['nameorg_check'] = data1['nameOrig'].astype(str).str[0]
print(data1.nameorg_check.unique().sum())

##nameDest ##
print(data1.nameDest.unique()) 
data1['namedest_check'] = data1['nameDest'].astype(str).str[0]
plt.hist(data1['namedest_check'], bins=2, color='skyblue', edgecolor='black')
plt.xlabel("Transaction Account")
plt.ylabel("TransactionAmount")
accounts_used_fewtimes = round(100 * data1.nameDest.nunique()/data1.nameDest.count(), 2)
print('Percentage of the accounts received more than one fraudulent transaction is {}%.'.format(accounts_used_fewtimes))



accounts_used_fewtimes = round(100 * data2.nameDest.nunique()/data2.nameDest.count(), 2)
print('Percentage of the accounts received more than one fraudulent transaction is {}%.'.format(accounts_used_fewtimes))



##plotting for fraud cases
plt.hist(data2['namedest_check'], bins=2, color='skyblue', edgecolor='black')
plt.xlabel("Transaction Account")
plt.ylabel("TransactionAmount")

###isFlaggedFraud##
fraud_ratio1 = round(100 * data1.isFraud.sum()/data1.isFraud.count(), 2)
print('Percentage of the fraud transactions is {}% out of all the transactions.'.format(fraud_ratio1))

fraudratio2 = 100*round(data1['isFlaggedFraud'][data1.isFlaggedFraud == 1].count()/data1['isFraud'][data1.isFraud == 1].count(),4)
print('Percentage of the fraud transactions that were flagged by business model is extremely low - {}%.'.format(fraudratio2))
data4 = not_outliers.copy()
#oldbalanceOrg#
print(data4['oldbalanceOrg'].count())
print(data4['oldbalanceOrg'].mean()) 
print(data4['oldbalanceOrg'].median())
#print(data1['amount'].mode())+-*
print(data4['oldbalanceOrg'].std())
#print(data1['amount'].value_counts())
#print(data['amount'].unique())
print(data4['oldbalanceOrg'].max())
#print(data['step'].min())


plt.hist(data4['oldbalanceOrg'], bins=15, color='skyblue', edgecolor='black')

plt.hist(data4['oldbalanceOrg'], bins=15, color='skyblue', edgecolor='black')

##checking for outlier##
boxplot = data4.boxplot(column= ['oldbalanceOrg'])
# IQR
Q1 = np.percentile(data4['oldbalanceOrg'], 25, method='midpoint')
Q3 = np.percentile(data4['oldbalanceOrg'], 75, method='midpoint')
IQR = Q3 - Q1
print(IQR)

#outliers = data1[((data1['amount']<(Q1-1.5*IQR)) | (data1['amount']>(Q3+1.5*IQR)))]
not_outliers_oldbalance = data4[~((data4['oldbalanceOrg']<(Q1-1.5*IQR)) | (data4['oldbalanceOrg']>(Q3+1.5*IQR)))]
not_outliers_oldbalance.describe()

data5 = data4.copy()
data4 = not_outliers_oldbalance.copy()
#newbalanceOrig#
print(data4['newbalanceOrig'].count())
print(data4['newbalanceOrig'].mean()) 
print(data4['newbalanceOrig'].median())
#print(data1['amount'].mode())
print(data4['newbalanceOrig'].std())
print(data1['newbalanceOrig'].value_counts())
#print(data['amount'].unique())
print(data4['newbalanceOrig'].max())
#print(data['step'].min())
perc2 = 100*round(data2['newbalanceOrig'][data2.newbalanceOrig != 0].count()/data2['newbalanceOrig'][data2.newbalanceOrig == 0].count(),3)
print('In general the majority of the Fraud transactions emptied the accounts completely. Percentage of the Fraud transactions that left something on the account of Origin is {}%'.format(perc2))
data1.columns
data1[['amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'isFraud']].corr()


plt.hist(data4['newbalanceOrig'], bins=15, color='skyblue', edgecolor='black')

plt.hist(data4['newbalanceOrig'], bins=15, color='skyblue', edgecolor='black')

##checking for outlier##
boxplot = data4.boxplot(column= ['newbalanceOrig'])
# IQR
Q1 = np.percentile(data4['newbalanceOrig'], 25, method='midpoint')
Q3 = np.percentile(data4['newbalanceOrig'], 75, method='midpoint')
IQR = Q3 - Q1
print(IQR)

#outliers = data1[((data1['amount']<(Q1-1.5*IQR)) | (data1['amount']>(Q3+1.5*IQR)))]
not_outliers_newbalanceOrig = data4[~((data4['newbalanceOrig']<(Q1-1.5*IQR)) | (data4['newbalanceOrig']>(Q3+1.5*IQR)))]
not_outliers_newbalanceOrig.describe()




#data2.to_csv(r'F:\step\Fraud_1.csv')

##Bar graph for target variable###
