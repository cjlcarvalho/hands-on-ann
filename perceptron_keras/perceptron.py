from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import numpy as np

def model():

    m = Sequential()
    m.add(Dense(1, input_dim=2, activation='sigmoid'))
    return m

m = model()
m.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

X = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
Y = np.array([1, 1, 1, 0])

m.fit(X, Y, batch_size=1, epochs=1000, verbose=1)

predictions = m.predict(np.array([[1, 0], [0, 0], [0, 1], [1, 1]]))

for i in predictions:
    print(round(i[0]))
