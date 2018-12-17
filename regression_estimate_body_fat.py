# Objective: estimate body fat with deep neural network

# Step 1: extract data
from pandas import read_csv

data = read_csv('https://www.openml.org/data/get_csv/52738/bodyfat.csv')

print(data.head())
dataset = data.values

X = dataset[:, 0:14].astype(float)
y = data[['class']].values.astype(float)

print('X: ', X.shape)
print('y: ', y.shape)

# divide data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

# Normalize data
from sklearn.preprocessing import StandardScaler
x_scaler = StandardScaler().fit(X_train)
tX_train = x_scaler.transform(X_train)
tX_test = x_scaler.transform(X_test)

print('tX_train: \n', tX_train[:5])
print('tX_test: \n', tX_test[:5])

from sklearn.preprocessing import MinMaxScaler
y_scaler = MinMaxScaler().fit(y_train)
ty_train = y_scaler.transform(y_train)
ty_test = y_scaler.transform(y_test)

print('ty_train: \n', ty_train[:5])
print('ty_test: \n', ty_test[:5])

num_classes = tX_train.shape[1]
num_out_classes = ty_train.shape[1]

print('num_classes: ', num_classes)
print('num_out_classes: ', num_out_classes)

# Step 2: build network
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(units=32,
               input_dim=num_classes,
               kernel_initializer='normal',
               activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32,
               kernel_initializer='normal',
               activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_out_classes))
model.compile(loss='mean_squared_error', optimizer='adam')

cb_early_stop = EarlyStopping(monitor='val_loss', patience=5)

print(model.summary())

# Step 3: train
history = model.fit(tX_train, 
                    ty_train, 
                    epochs=100,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[cb_early_stop])

# verify
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss func')
plt.ylabel('loss/val_loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

scores = model.evaluate(tX_test, ty_test)
print('\n%s: %.3f' % (model.metrics_names, scores))

# Step 4: test

y_pred = model.predict(tX_test)
y_pred = y_scaler.inverse_transform(y_pred)
print('y_pred: \n', y_pred[:5])

import pandas as pd
dfy = pd.DataFrame({'Truth class': y_test[...,0],
                  'Pred class': y_pred[...,0]})

long_dfy = pd.melt(dfy, value_vars=['Truth class', 'Pred class'])

from plotnine import *
(ggplot(long_dfy, aes(x='value', color='variable', fill='variable'))
  + geom_density(alpha=0.5)
  + theme_bw())
