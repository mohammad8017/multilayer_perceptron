from keras import models
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import keras.losses
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def normalize(data):
    tmp = [((item - min(data)) / ((max(data)-min(data)))) for item in data]
    return np.array(tmp)


data = pd.read_csv('ADANIPORTS.csv')
closePrice , openPrice, low, high, volume = data['Close'].to_numpy(dtype=float), data['Open'].to_numpy(dtype=float), data['Low'].to_numpy(dtype=float), data['High'].to_numpy(dtype=float), data['Volume'].to_numpy(dtype=float)

closePrice , openPrice, low, high, volume = np.delete(closePrice, -1), np.delete(openPrice, -1), np.delete(low, -1), np.delete(high, -1), np.delete(volume, -1)
nextClose = (data['Close'].to_numpy(dtype=float))
nextClose = np.delete(nextClose, 0)

closePrice, openPrice, low, high, volume, nextClose = normalize(closePrice), normalize(openPrice), normalize(low), normalize(high), normalize(volume), normalize(nextClose)

features = []
for i in range(len(closePrice)):
    features.append([closePrice[i], openPrice[i], low[i], high[i], volume[i]])

res = []
for i in range(len(closePrice)):
    if nextClose[i] >= closePrice[i]: res.append(1)
    else: res.append(0)

print(features[:10])

'''
xTrain, xTest, yTrain, yTest = train_test_split(features, res, test_size=0.1)
print(len(xTrain))


model = keras.Sequential()

model.add(layers.Dense(100, input_shape=(5,)))
model.add(layers.Dense(50))
model.add(layers.Dense(10))
model.add(layers.Dense(5))
model.add(layers.Dense(1, activation='softmax'))

model.compile(optimizer='Adam', loss=keras.losses.BinaryCrossentropy())
model.summary()
resss = model.fit(xTrain, yTrain, batch_size=64, epochs=100, validation_data=(xTest, yTest))

loss = model.evaluate(xTest, yTest)

# plt.plot(resss.history['loss'], label='train')
# plt.plot(resss.history['val_loss'], label='val')
# plt.show()

pred = model.predict(xTest)
print('===========================')
print('pred    org')
for i in range(len(pred)):
    print(pred[i], end='\t')
    print(yTest[i])



# print('===========================')
# print(r2_score(yTest, pred))
'''