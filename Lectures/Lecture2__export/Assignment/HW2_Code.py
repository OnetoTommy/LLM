import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# In the weather dataset with ‘RainTomorrow’ as label and all other attributes as features,
# split the data into training (75%) and test (25%).
# Train a Feed Forward Neural on train dataset with ‘RainTomorrow’ as label
# and all other attributes as features with two different designs:
# 1.One hidden layer with 8 units
# 2.Two hidden layers with 8 and 5 units
# Calculate accuracy of both models on test data for each of the following settings:
# batch size = 32, epochs = 10, LearningRate = 0.0001
# batch size = 4, epochs = 30, LearningRate = 0.01

# Import the weather data
df = pd.read_csv('weather.csv')
# print(df)

# split the features and label for data
feat_df = df.loc[:, df.columns != 'RainTomorrow']
label_df = df.loc[:, df.columns == 'RainTomorrow']

#Normalize Data
norm_feat_df = (feat_df - feat_df.mean()) / feat_df.std()
# print(norm_feat_df)

# Split the data into training (75%) and test (25%)
x_train, x_test, y_train, y_test = train_test_split(norm_feat_df, label_df, test_size=0.25)

# Define the One hidden layer with 8 units
def hidden_layer(i, j):
    nn = Sequential()
    if i:
        nn.add(Dense(units=i, activation='relu'))
    if j:
        nn.add(Dense(units=j, activation='relu'))
    nn.add(Dense(units=1, activation='sigmoid'))
    return nn

# Calculate accuracy of both models on test data
def para_nn(nn, lr, size, epochs, x_train, y_train):
    nn.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(x_train, y_train, batch_size=size, epochs=epochs)

    arr_layer= [(8, 0), (8, 5)]
    arr_nn = [[0.0001, 32, 10], [0.01, 4, 30]]
res = []
for i, j in arr_layer:
    for lr, size, epoch in arr_nn:
        nn = hidden_layer(i, j)
        para_nn(nn, lr, size, epoch, x_train, y_train)
        loss, accuracy = nn.evaluate(x_test, y_test)
        res.append((loss, accuracy))
print(res)

losses = [x[0] for x in res]
accs = [x[1] for x in res]
x = range(1, len(res) + 1)
plt.figure(figsize=(8, 5))
plt.plot(x, losses, marker="o", label="Loss")
plt.plot(x, accs, marker="s", label="Accuracy")
plt.xlabel("Experiment Index")
plt.ylabel("Value")
plt.title("Loss & Accuracy Comparison")
