# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:18:01 2019

@author: sarth
"""
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping into 3D array (by adding 1 more dimension or indicator that 
# also would affect the stock price of google)

# Since the LSTM would take 3D array we have to convert the X_train to a 3D
# array by using reshape function 

# Here reshape takes np.reshape(rows, columns, number of indicators)
# X_train.shape gives (rows,columns) so X_train.shape[0] gives number of rows
# and X_train.shape[1] gives number of columns

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Building the RNN
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout #to add dropout regularization

#Initializing the LSTM (stacked LSTM with dropout)

#creating a regressor as a sequence of layers which gives a continous output 
#rather than classifying them based on categories:
regressor = Sequential()

# Adding the first LSTM layer and adding dropout regularization to prevent 
# overfitting
# We'd add 3 arguments here: 1) Number of LSTM cells. Here I took 50 neurons 
# which will give a model with high dimensionality to capture the upward and 
# and downward trend 2) We are going to add more LSTM layers, hence return 
# sequences = True (it is false by default) 3) Input shape : it contains the 
# last 2 steps in reshape corresponding to timestep and indicators; so just
# copy and paste them in between new paranthesis

regressor.add(LSTM(units = 50, return_sequences = True, 
                   input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding a 2nd LSTM layer with dropout
#Here we can skip input shape as it is automatically recognized in subsequent
#LSTM layers
#keeping the neurons at 50 ensures that the model has high dimensionality and 
# high complexity thus giving better results
regressor.add(LSTM (units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) #for regularization

#3rd LSTM layer
regressor.add(LSTM (units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# In 4th LSTM layer we have to set the return_sequence to False as we dont have
# to return any more sequences. Since thats' the default value we can remove it 
regressor.add(LSTM (units = 50))
regressor.add(Dropout(0.2))

#Adding the output layer by using Dense
regressor.add(Dense(units = 1))

#Compiling using optimizer (RMS prop is generally a good optimizer for RNN)
# But Adam Optimizer is powerful and universally prefered as it updates weights
# with relevant values
#loss for regression type of problem (continous output) is rms error
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
# in regressor.fit() we have to pass 4 arguments
regressor.fit(X_train, y_train, epochs =100, batch_size = 32)

# Making Predictions and viewing the results
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
testing_set = dataset_test.iloc[:, 1:2].values
real_stock_price = testing_set

#Obtaining the predicted stock of 2017 
# To obtain the stock price for each ith day in january 2017 we need to have
# stock price of 60 days before january
#for horizontal concatenate axis = 1 and for vertical = 0
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), 
                          axis = 0)
#suppose we want to find prediction for january 3rd. then we have to select 60
#days before january 3rd which would be input 
#lower bound is 60 days before the start of test dataset
#upper bound is the stock price before the last financial day
inputs = dataset_total[len(dataset_total) - len(dataset_test) -60:].values
# ^ .values added to make it numpy array
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualizing the result
plt.plot(real_stock_price, color = 'red', label = 'Actual Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend
plt.show()
    