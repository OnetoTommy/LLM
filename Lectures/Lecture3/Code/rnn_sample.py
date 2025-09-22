import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

#Read data
dataset_train = pd.read_csv('passengers_ds.csv')
train = dataset_train.loc[:, ['total_passengers']].values

#Normalize data
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train)

#Create Sequence with Length of 50
X_train = []
y_train = []
timesteps = 50
for i in range(timesteps, 144):
    X_train.append(train_scaled[i - timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshape data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Create RNN model
rnn = Sequential()
rnn.add(SimpleRNN(units = 50, activation='relu'))
rnn.add(Dense(units = 1))

#Set optimizer and loss
rnn.compile(optimizer='adam', loss='mean_squared_error')

#Fitting the RNN to the Training set
rnn.fit(X_train, y_train, epochs=100, batch_size=32)


#Plot Real vs Predicted

# Predict values based on the RNN model
predictions = rnn.predict(X_train)
predictions_rescaled = scaler.inverse_transform(predictions)
y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(y_train_rescaled, label='Actual')
plt.plot(predictions_rescaled, label='Predicted', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend()
plt.show()
