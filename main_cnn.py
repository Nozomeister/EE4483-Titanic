import numpy as np
import scipy
import pandas as pd
from sklearn import preprocessing
import keras
import tensorflow as tf
from keras.models import Sequential
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
dataset = pd.read_csv('./data/train.csv')
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



dataset.info()
export_csv = dataset.to_csv (r"/home/students/student3_5/option2/processing_cnn.csv", index = None, header=True)

#end of processing data, seperate dataset into output and input
y = dataset.Survived
#dropping Ticket number and PassengerId
# X = dataset.drop(['Survived','SibSp', 'Parch', 'Embarked', 'CabinName', 'FamilySize', 'Title'], axis = 1)
X = dataset.drop(['Survived','SibSp'], axis = 1)

#end of cleaning data, now we transform data to CNN input

sc = preprocessing.StandardScaler()
X = sc.fit_transform(X)
y_train_onehot = pd.get_dummies(y).values

img_rows, img_cols = 3, 3

pool_size = (1, 1)
kernel_size = (1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y_train_onehot, test_size = 0.2)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
#X_train = X.reshape(X.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

tf.keras.backend.clear_session()
keras.backend.clear_session()
#model
classifier = Sequential()
classifier.add(Convolution2D(32, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
classifier.add(Activation('relu'))
classifier.add(Convolution2D(64, kernel_size[0], kernel_size[1]))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=pool_size))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(512))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(2))
classifier.add(Activation('sigmoid'))
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )

classifier.fit(X_train, y_train, validation_data = (X_val, y_val),  batch_size = 5, nb_epoch = 100)

classifier.save("model_cnn.h5")