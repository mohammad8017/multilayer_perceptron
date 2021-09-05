import random as rnd
import numpy as np
from numpy.testing._private.utils import print_assert_equal
import pandas as pd
import math
#import matplotlib.pyplot as plt

def normalize(data):
    return [((item - min(data)) / ((max(data)-min(data)))) for item in data]

def findSlope(res, b, W, X):
    return -2 * (res - (b + np.dot(np.transpose(W), X)))

def prediction(weight, bias, data):
    tmp = np.dot(np.array(weight), np.array(data))
    
    return 0 if tmp + bias < 0.0 else 1

allData = pd.read_csv('ADANIPORTS.csv')
closee, openn = allData['Close'], allData['Open'][:500]
closee, openn = closee.tolist(), openn.tolist()
tomorowClose = closee[1:501]
closee = closee[:500]
closee, openn, tomorowClose = normalize(closee), normalize(openn), normalize(tomorowClose)


data = []
res = []
for i in range(500):
    # tmp = []
    # tmp.append(closee[i])
    # tmp.append(openn[i])
    data.append([closee[i], openn[i]])
    if tomorowClose[i] > closee[i] :
        res.append(1)
    else:
        res.append(0)

learningRate = 0.1
weight = [0.2, -0.35]
bias = 0.23
weights = [0.0 for i in range(len(data[0]))]
for epoche in range(500):
    sumError = 0.0
    for row in range(len(data)):
        weight_T = np.transpose(weight)
        pred = prediction(weight_T, bias, data[row])

        sumError += (res[row] - pred)**2

        for i in range(len(weight)):
            weight[i] += learningRate * (res[row] - pred) * data[row][i]
        bias += learningRate * (res[row] - pred)
    print('epoche', epoche, end='\t')
    print('sum error = {0:.3f}'.format(sumError))

print('weight:', weight)

    # print('\t\tW_T:', weight_T, '\t\tbias:', bias)
    # for i in range(len(data)):
    #     print('data:',*data[i], '\t\t result:',pred[i])
biasOld = bias
while bias >= biasOld:
    slope = findSlope(res[50], bias, weight, data[50]) + findSlope(res[70], bias, weight, data[70]) + findSlope(res[100], bias, weight, data[100])

    stepSize = slope * learningRate
    biasOld = bias
    bias = biasOld - stepSize

    if 1 - stepSize < 0.0009 or 1 - stepSize <1.0009:
        break 

print('bias:', bias)
print('step size:', stepSize)
print('slope:', slope)


# for i in range(len(data)):
#     if findSlope(res[i], bias, weight, data[i]) == slope:
#         print('data of index', i)
#         print(data[i])
#         exit()