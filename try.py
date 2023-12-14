import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd

df = pd.read_csv("eg 50-55-small.csv")

y = df['lable']
del df['lable']
x = df.values

x_train = x[:500000]
y_train = y[:500000]

x_valid = x[500000:510000]
y_valid = y[500000:510000]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

y_train = y_train / 64
y_valid = y_valid / 64

model = Sequential()
model.add(Dense(units = 512, activation='relu', input_shape=(64,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = 1))

model.compile(loss='mean_squared_error', metrics=['mae'])

print(model.summary())

model.fit(x_train, y_train, batch_size=16, epochs=10,verbose=1, validation_data=(x_valid, y_valid))

model.save('model-50-55-small')