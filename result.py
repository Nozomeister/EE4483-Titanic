import numpy as np
import scipy
import pandas as pd
from sklearn import preprocessing
# import keras
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from keras.utils import plot_model
#processing function
#extract cabin code
def DeckNum(Code):
    if pd.isnull(Code):
        category = 'Unknown'
    else:
        category = Code[0]
    return category
#filling missing age in dataset
def ReplaceAge(means, dataset, title_list):
    for title in title_list:
        temp = dataset['Title'] == title
        dataset.loc[temp, 'Age'] = dataset.loc[temp, 'Age'].fillna(means[title])

#---Preprocessing Data-----------
dataset = pd.read_csv('./reg_result.csv')
# dataset.info()
# print(dataset['Pclass'].describe())
#print(dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()) #Finding correlation between Pclass and Survived

#dropping Ticket number and PassengerId
dataset.drop(['Ticket','PassengerId'], 1, inplace=True)

# #extract Cabin name


CabinName = np.array([DeckNum(cabin) for cabin in dataset['Cabin'].values])
#append to the dataset
dataset = dataset.assign(CabinName = CabinName)


#combine Parch and Sibsp into a family column
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# print(dataset['FamilySize'].value_counts())
# print(dataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
#extract title from Name then drop it

dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')

dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

# print(dataset['Title'].value_counts())
# print(dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
#print(dataset[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

# #drop Name and Cabin after extract CabinName and Title
dataset.drop(['Name', 'Cabin'], 1, inplace = True)

# #label the embarked and fill in NaN value with highest one - S
dataset['Embarked'] = dataset['Embarked'].fillna('S')
# print(dataset[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

#taking mean value of age in each group to replace missing age
means = dataset.groupby('Title')['Age'].mean()
title_list = ['Master','Miss','Mr','Mrs','Others']
ReplaceAge(means, dataset, title_list)
# print(dataset['Age'].describe())
# print(dataset['Fare'].describe())

print(dataset['Survived'].value_counts())
print(dataset[['Sex', 'Survived']].groupby(['Sex'], as_index=False))
# print(dataset[dataset['Age'] >18].groupby(['Survived'], as_index = False).value_counts())
# print(dataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).counts())
print(dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean())
print(dataset[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean())

#labelencoder categorical features
labelencoder = preprocessing.LabelEncoder()
dataset['CabinName'] = labelencoder.fit_transform(dataset['CabinName'])
dataset['Embarked'] = labelencoder.fit_transform(dataset['Embarked'])
dataset['Sex'] = labelencoder.fit_transform(dataset['Sex'])
dataset['Title'] = labelencoder.fit_transform(dataset['Title'])

