import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


#Input Dateset
org_df = pd.read_csv("Liver_disease_data.csv")

#Define features and label for KNN
label_df =  org_df.loc[:,org_df.columns == 'Diagnosis']
feat_df =  org_df.loc[:,org_df.columns != 'Diagnosis']

##Encoding Categorical Variables
feat_df= pd.get_dummies(feat_df, dtype='int')

#Normalize Data
norm_feat_df = (feat_df - feat_df.mean()) / feat_df.std()

#Seperate test and train data
train_feat,test_feat,train_lbl,test_lbl = train_test_split(norm_feat_df,label_df,test_size=0.2)

#KNN Model
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(train_feat, train_lbl)

# accuracy measures
pred_y = knn.predict(test_feat)
accuracy=accuracy_score(pred_y,test_lbl)
print(accuracy)


#kmeans
model = KMeans(n_clusters=2)
model.fit(norm_feat_df)

first_cluster = feat_df.loc[model.labels_ == 0,:]
second_cluster = feat_df.loc[model.labels_ == 1,:]

# Plotting the results
plt.scatter(first_cluster.loc[:, 'AlcoholConsumption'], first_cluster.loc[:, 'LiverFunctionTest'], color='red')
plt.scatter(second_cluster.loc[:, 'AlcoholConsumption'], second_cluster.loc[:, 'LiverFunctionTest'], color='blue')
plt.show()