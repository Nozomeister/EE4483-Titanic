import numpy as np
import scipy
import pandas as pd
from sklearn import preprocessing
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
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
export_csv = dataset.to_csv (r"/home/students/student3_5/option2/processing.csv", index = None, header=True)

#end of processing data, seperate dataset into output and input
y = dataset['Survived'].values
dataset = dataset.drop(['Survived', 'Parch', 'SibSp'], axis = 1)

X = dataset.values
sc = preprocessing.StandardScaler()
X = sc.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
tf.keras.backend.clear_session()
keras.backend.clear_session()
classifier = Sequential()
#input layer
classifier.add(Dense(255 , activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 8))
#hidden layer
classifier.add(Dense(25, activation = 'relu', kernel_initializer = 'random_uniform'))

# classifier.add(Dense(25, activation = 'relu', kernel_initializer = 'random_uniform'))
classifier.add(Dense(25, activation = 'relu', kernel_initializer = 'random_uniform'))

classifier.add(Dense(25, activation = 'relu', kernel_initializer = 'random_uniform'))

classifier.add(Dense(10, activation = 'relu', kernel_initializer = 'random_uniform'))
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_uniform'))
#compiling NN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
history = classifier.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = 5, nb_epoch = 100, callbacks = [es_callback])

classifier.save("model_6.5layers.h5")