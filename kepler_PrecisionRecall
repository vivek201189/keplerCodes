'''This is an example for understanding precision and recall
for Kepler'''

import numpy as np

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import containers
from keras.layers.noise import GaussianNoise
from keras.layers.core import Dense, AutoEncoder
from keras.utils import np_utils
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score)

np.random.seed(1337)

max_len = 100
max_words = 20000
batch_size = 64
nb_classes = 2
nb_epoch = 2

(X_train, y_train), (X_test, y_test) = \
    imdb.load_data(nb_words=max_words, test_split=0.2)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('Train: {}'.format(X_train.shape))
print('Test: {}'.format(X_test.shape))

ae = Sequential()

encoder = containers.Sequential([
    GaussianNoise(0.5, input_shape=(100,)),
    Dense(input_dim=100, output_dim=80, activation='sigmoid',
          init='uniform'),
])
decoder = Dense(input_dim=80, output_dim=100, activation='sigmoid')
ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                   output_reconstruction=False))

ae.compile(loss='mean_squared_error', optimizer='sgd')

ae.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch)

model = Sequential()
model.add(ae.layers[0].encoder)
model.add(Dense(input_dim=80, output_dim=nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(
    X_train, Y_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    show_accuracy=True,
    validation_data=(X_test, Y_test),
)

y_pred = model.predict_classes(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy: {}'.format(accuracy))
print('Recall: {}'.format(recall))
print('Precision: {}'.format(precision))
print('F1: {}'.format(f1))
