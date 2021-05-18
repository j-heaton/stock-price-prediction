import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import timedelta


def split_dataset(dataset):
    #split into train and test dataset
    training_data_len = math.ceil(len(dataset)*.75)
    train_data = dataset[0:training_data_len]
    test_data = dataset[training_data_len-60:]
    
    #scale the data
    train_scaler = MinMaxScaler(feature_range=(0,1))
    test_scaler = MinMaxScaler()
    scaled_train = train_scaler.fit_transform(train_data)
    scaled_test = test_scaler.fit_transform(test_data)
    
    #reshape training data for LSTM
    X_train = []
    y_train = []
    for i in range(60, len(train_data)):
        X_train.append(scaled_train[i-60:i])
        y_train.append(scaled_train[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    #reshape test data for LSTM
    X_test = []
    y_test = []
    for i in range(60, test_data.shape[0]):
        X_test.append(scaled_test[i-60:i])
        y_test.append(scaled_test[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    return [X_test, y_test, X_train, y_train, test_scaler]
    
#LSTM model
def make_model(X_train):
    model = Sequential()

    model.add(LSTM(units = 40, activation = 'linear',return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 40, activation = 'linear',return_sequences = True))
    model.add(Dropout(0.2))
    

    model.add(LSTM(units = 64, activation = 'linear'))
    model.add(Dropout(0.2))
    
    model.add(Dense(10))
    model.add(Dense(1))
    
    
    #compile the model
    model.compile(optimizer='adam', loss = 'mean_squared_error')
    return model
    
def get_yhat(model, X_test, y_test, test_scaler, dataset):
    #extract the index from the dataframe (since the test data is in a np.array it no longer has a datetime index)
    test_index = dataset.iloc[-y_test.shape[0]:].index
    test_index = test_index+timedelta(1)
    y_test_reshaped = np.reshape(y_test,(y_test.shape[0], 1))
    #unscale
    test_unscaler = MinMaxScaler()
    test_unscaler.min_, test_unscaler.scale_ = test_scaler.min_[0], test_scaler.scale_[0]
    y_test_unscaled = test_unscaler.inverse_transform(y_test_reshaped)
    y_test_df = pd.DataFrame(data=y_test_unscaled, columns=['Close']).set_index(test_index)
    #use model to make predictions
    yhat = model.predict(X_test)
    yhat_unscaled = test_unscaler.inverse_transform(yhat)
    yhat_df = pd.DataFrame(data=yhat_unscaled, columns=['Close']).set_index(test_index)
    return [yhat ,yhat_df, y_test_df]