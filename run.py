# ==============================================================
#   Copyright (C) 2021 whubaichuan. All rights reserved.
#   function： Demo of Vessel Trajectory Prediction by sequence-to-sequence model (LSTM)
# ==============================================================
#   Create by whubaichuan at 2021.05.02
#   Version 1.0
#   whubaichuan [huangbaichuan@whu.edu.cn]
# ==============================================================


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import matplotlib 
import glob, os
import seaborn as sns
import sys
from sklearn.preprocessing import MinMaxScaler


path="C:/Users/work_computer/Desktop/Vessel_Trajectory_Prediction/"

columns = ['X','Y']
data = pd.read_csv(path+"data/AIS.csv",  names=columns)
print(data)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[['X','Y']].values)

reframed = series_to_supervised(scaled_data, 20, 1)


print(reframed.info())


train_days = 300
valid_days = 60
values = reframed.values
train = values[:train_days, :]
valid = values[train_days:train_days+valid_days, :]
test = values[train_days+valid_days:, :]
train_X, train_y = train[:, :-2], train[:, -2:]
valid_X, valid_y = valid[:, :-2], valid[:, -2:]
test_X, test_y = test[:, :-2], test[:, -2:]


train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape, test_X.shape, test_y.shape)


model1 = Sequential()
model1.add(LSTM(50, activation='relu',input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
#model1.add(Dense(2, activation='linear'))
model1.add(LSTM(2, activation='relu'))
model1.compile(loss='mean_squared_error', optimizer='adam') 
# fit network
LSTM = model1.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(valid_X, valid_y), verbose=2, shuffle=False)


# plt.plot(LSTM.history['loss'], label='train')
# plt.plot(LSTM.history['val_loss'], label='valid')
# plt.legend()
# plt.show()

plt.figure(figsize=(24,8))
plt.xlabel('x')
plt.ylabel('y')
train_predict = model1.predict(train_X)
valid_predict = model1.predict(valid_X)
test_predict = model1.predict(test_X)

#for dense
# plt.plot(values[:,0],values[:,1], label='raw_trajectory',c='b')
# plt.plot(train_predict[:,0,0], train_predict[:,0,1],label='train_predict', c='g')
# plt.plot(valid_predict[:,0,0], valid_predict[:,0,1],label='valid_predict', c='y')
# plt.plot(test_predict[:,0,0], test_predict[:,0,1], label='test_predict', c='r')
# plt.legend()
# plt.show()


#for LSTM
plt.plot(values[:,0],values[:,1], label='raw_trajectory',c='b')
plt.plot(train_predict[:,0], train_predict[:,1],label='train_predict', c='g')
plt.plot(valid_predict[:,0], valid_predict[:,1],label='valid_predict', c='y')
plt.plot(test_predict[:,0], test_predict[:,1], label='test_predict', c='r')
plt.legend()
plt.show()
