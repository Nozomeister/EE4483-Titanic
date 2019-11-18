import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import os

batch_size = 32
epochs = 70
num_predictions = 20
model_name = 'cifar10.h5'
num_classes = 10
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert output vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

tf.keras.backend.clear_session()
keras.backend.clear_session()

classifier = Sequential()

# classifier.add(Conv2D(32,(3, 3), padding = 'same',
#                         input_shape=x_train.shape[1:]))
# classifier.add(Activation('relu'))
# classifier.add(Conv2D(32,(3, 3)))
# classifier.add(Activation('relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.25))

classifier.add(Conv2D(64,(3, 3), padding = 'same', input_shape=x_train.shape[1:])) #first Conv block
classifier.add(Activation('relu'))
classifier.add(Conv2D(64,(3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(128,(3, 3), padding = 'same')) #2nd Conv block
classifier.add(Activation('relu'))
classifier.add(Conv2D(128,(3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(256,(3, 3), padding = 'same')) #3rd Conv block
classifier.add(Activation('relu'))
classifier.add(Conv2D(256,(3, 3)))
classifier.add(Activation('relu'))
classifier.add(Conv2D(256,(3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.4))


classifier.add(Conv2D(512,(3, 3), padding = 'same')) #4th Conv block
classifier.add(Activation('relu'))
classifier.add(Conv2D(512,(3, 3)))
classifier.add(Activation('relu'))
classifier.add(Conv2D(512,(3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.4))

classifier.add(Conv2D(512,(3, 3), padding = 'same')) #5th Conv block
classifier.add(Activation('relu'))
classifier.add(Conv2D(512,(3, 3)))
classifier.add(Activation('relu'))
classifier.add(Conv2D(512,(3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.4))

classifier.add(Flatten())

classifier.add(Dense(4096))
classifier.add(Activation('relu')) #1st FCN
classifier.add(Dropout(0.5))

classifier.add(Dense(4096))
classifier.add(Activation('relu')) #2nd FCN
classifier.add(Dropout(0.5))

classifier.add(Dense(10))
classifier.add(Activation('softmax'))

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
classifier.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test), shuffle = True,  callbacks = [es_callback])

classifier.save("model_cifar10.h5")

scores = classifier.evaluate(x_test, y_test, verbose = 1)
print('Test loss: ', scores[0])
print('Test accuracy: ', scores[1])