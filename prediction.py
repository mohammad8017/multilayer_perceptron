import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import keras.losses
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


'''funnction for normalize values of lists'''
def normalize(data):
    return np.array([((item - min(data)) / ((max(data)-min(data)))) for item in data])


'''read data from file and normalization'''
data = pd.read_csv('ADANIPORTS.csv')
closePrice, volume = data['Close'].to_numpy(dtype=float), data['Volume'].to_numpy(dtype=float)

closePrice, volume = np.delete(closePrice, -1), np.delete(volume, -1)
nextClose = (data['Close'].to_numpy(dtype=float))
nextClose = np.delete(nextClose, 0)

closePrice, volume, nextClose = normalize(closePrice), normalize(volume), normalize(nextClose)

'''collect close price and volume of last 30 days'''
compact = [[p, v] for p, v in zip(closePrice, volume)]
last30Days = []
nextPrice = []
for i in range(30, len(closePrice)-1):
    last30Days.append(compact[i-30:i])
    nextPrice.append([closePrice[i]])  


'''split datas to train and set lists'''
xTrain, xTest, yTrain, yTest = train_test_split(last30Days, nextPrice, test_size=0.1, shuffle=False)


'''create multi layer perceptron for training model'''
model = keras.Sequential()
model.add(layers.Flatten(input_shape=(30,2)))
model.add(layers.Dense(100))
model.add(layers.Dense(50))
model.add(layers.Dense(10))
model.add(layers.Dense(5))
model.add(layers.Dense(1))


'''compile model and give summary of our model'''
model.compile(optimizer='Adam', loss='mse')
model.summary()



'''fit and evaluate our model'''
resss = model.fit(xTrain, yTrain, batch_size=64, epochs=100)
model.evaluate(xTest, yTest)

'''predict value of test set'''
pred = model.predict(xTest)
print('===========================')
print('pred    org')
for i in range(len(pred)):
    print(pred[i], end='\t')
    print(yTest[i])


'''plot predicted value and orignal value of test set'''
plt.plot(pred, 'C1--', linewidth=0.5)
plt.plot(yTest, 'C0', linewidth=0.7)
plt.show()