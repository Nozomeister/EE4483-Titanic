import numpy as np
import scipy
import pandas as pd
from sklearn import preprocessing
import keras
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import  Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from keras.utils import plot_model

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
dataset = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/submission.csv')
#dropping Ticket number and PassengerId
dataset.drop(['Ticket','PassengerId'], 1, inplace=True)

#extract Cabin name


CabinName = np.array([DeckNum(cabin) for cabin in dataset['Cabin'].values])
#append to the dataset
dataset = dataset.assign(CabinName = CabinName)

#combine Parch and Sibsp into a family column
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#extract title from Name then drop it
dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
#df = dataset['Title']

dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')

dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

#drop Name and Cabin after extract CabinName and Title
dataset.drop(['Name', 'Cabin'], 1, inplace = True)

#label the embarked and fill in NaN value with highest one - S
dataset['Embarked'] = dataset['Embarked'].fillna('S')

#taking mean value of age in each group to replace missing age
means = dataset.groupby('Title')['Age'].mean()
title_list = ['Master','Miss','Mr','Mrs','Others']
ReplaceAge(means, dataset, title_list)

#labelencoder categorical features
labelencoder = preprocessing.LabelEncoder()
dataset['CabinName'] = labelencoder.fit_transform(dataset['CabinName'])
dataset['Embarked'] = labelencoder.fit_transform(dataset['Embarked'])
dataset['Sex'] = labelencoder.fit_transform(dataset['Sex'])
dataset['Title'] = labelencoder.fit_transform(dataset['Title'])
dataset['Fare'].fillna((dataset['Fare'].mean()), inplace = True)
dataset.info()
export_csv = dataset.to_csv (r"/home/students/student3_5/option2/processing_test_cnn.csv", index = None, header=True)

#end of processing data, seperate dataset into output and input
#dropping Ticket number and PassengerId
# X = dataset.drop(['SibSp', 'Parch', 'Embarked', 'CabinName', 'FamilySize', 'Title'], axis = 1)
X = dataset.drop(['SibSp'], axis = 1)

#end of cleaning data, now we transform data to CNN input

sc = preprocessing.StandardScaler()
X = sc.fit_transform(X)

img_rows, img_cols = 3, 3
nb_filters = 1000
pool_size = (1, 1)
kernel_size = (1, 1)

X = X.reshape(X.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

tf.keras.backend.clear_session()
keras.backend.clear_session()
model = load_model("model_cnn.h5")
model.summary()
test = model.predict(X)
submission["Survived"] = test
submission.to_csv(r"/home/students/student3_5/option2/submission_cnn.csv", index = None, header=True)



test = model.predict(X)