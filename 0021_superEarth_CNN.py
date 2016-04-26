from __future__ import print_function
import numpy as np
from numpy import newaxis
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D

# set parameters:
batch_size = 32
input_length=500
nb_epoch = 100

print('Loading data...')
data_file1 = "/home/vivek/x_0021superearth_3days.txt"
data_file3 = "/home/vivek/y_0021superearth_3days.txt"

mega = np.loadtxt(data_file1, delimiter=',')
#print('this is mega.shape', mega.shape) #this is (1000, 121064)

X_train = mega[0:500, 0:60532]
#print ('shae of X_train', X_train.shape)
X_train = np.transpose(X_train)
#print ('shae of X_train after transpose', X_train.shape)
X_train = X_train[:, :, newaxis]
#print ('shae of X_train after transpose and newaxis', X_train.shape)

X_test = mega[500:1000,60532:121064]
#print ('shae of X_test', X_test.shape)
X_test = np.transpose(X_test)
#print ('shae of X_test after transpose', X_test.shape)
X_test = X_test[:, :, newaxis]
#print ('shae of X_test after transpose and newaxis', X_test.shape)

labels = np.loadtxt(data_file3, delimiter=',')
y_train = labels[:60532]
y_train = y_train[:,newaxis]
y_test = labels[60532:121064]
y_test = y_test[:,newaxis]

#print ('shae of y_train', y_train.shape)
#print ('shae of y test', y_test.shape)

print('Build model...')
model = Sequential()

model.add(Convolution1D(nb_filter=64,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu',
            input_dim=1, input_length=input_length)) 
#model.add(ZeroPadding1D(padding=1))                       
model.add(Convolution1D(nb_filter=64,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
                        
model.add(MaxPooling1D(pool_length=2))

#model.add(ZeroPadding1D(padding=1))
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
#model.add(ZeroPadding1D(padding=1))
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))

model.add(MaxPooling1D(pool_length=2))

#model.add(ZeroPadding1D(padding=1))                        
model.add(Convolution1D(nb_filter=256,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
#model.add(ZeroPadding1D(padding=1))
model.add(Convolution1D(nb_filter=256,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))

model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(nb_filter=512,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
model.add(Convolution1D(nb_filter=512,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))

# We flatten the output of the conv layer,
model.add(Flatten())

#two FC layers
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Dense(1024))
model.add(Dropout(0.50))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1)) 
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)
