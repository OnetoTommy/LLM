import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

#Input Dateset
org_df = pd.read_csv("heart_attack.csv")

#Labels and Features
label_df = org_df.loc[:,org_df.columns == 'heart_attack']
feat_df = org_df.loc[:,org_df.columns != 'heart_attack']

##Encoding Categorical Variables
feat_df= pd.get_dummies(feat_df, dtype='int')

#Normalize Features
feat_df = (feat_df - feat_df.mean()) / feat_df.std()

#Split Train and Test Data
x_train, x_test, y_train, y_test = train_test_split(feat_df, label_df, test_size = 0.3)

#Create NN Model
nn = Sequential()
# nn.add(Input(shape=(6,)))
nn.add(Dense(units=4,  activation='relu'))
nn.add(Dense(units=1,  activation='sigmoid'))

# Set Optimizer, Loss Function, and metric
nn.compile(optimizer=Adam(learning_rate=0.001) , loss='binary_crossentropy', metrics=['accuracy'])

#Train NN Model
nn.fit(x_train, y_train, batch_size=8, epochs=50)
print(nn.summary())

#Accuracy of Model on Test data
loss,accuracy = nn.evaluate(x_test,y_test)
print('accuracy=',accuracy,' , loss=',loss)



