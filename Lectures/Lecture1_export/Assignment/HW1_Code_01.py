import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# Input the CSV file by panadas
org_df = pd.read_csv('Heart_Failure.csv')

# Split the feature and label
label_df = org_df.loc[:, org_df.columns == 'HeartDisease']
feat_df = org_df.loc[:, org_df.columns != 'HeartDisease']

#Normalize Data
norm_feat_df = (feat_df - feat_df.mean()) / feat_df.std()

# Split data into training (72%), validation (8%) and test (20%)
t_feat, test_feat, t_label, test_label = train_test_split(norm_feat_df,label_df,test_size=0.2)
train_feat, val_feat, train_label, val_label = train_test_split(t_feat,t_label,test_size=0.1)

# Define the KNN model
def knn_model(k, train_feat, train_label, val_feat, val_label):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_feat, train_label)
    val_pre = knn.predict(val_feat)
    train_pre = knn.predict(train_feat)
    acc_val = accuracy_score(val_pre, val_label)
    acc_train = accuracy_score(train_pre, train_label)
    return acc_val, acc_train

KNN_model = [3, 9, 21]
acc_val_list = []
acc_train_list = []
for k in KNN_model:
    acc_val, acc_train = knn_model(k, train_feat, train_label, val_feat, val_label)
    acc_val_list.append(acc_val)
    acc_train_list.append(acc_train)

# Graph
plt.plot(KNN_model, acc_val_list, marker='o', label='Validation Accuracy')
plt.plot(KNN_model, acc_train_list, marker='s', label='Training Accuracy')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# KNN result for the best model
best_k = 0
best_acc = 0
for x, y in zip(KNN_model, acc_val_list):
    if y > best_acc:
        best_k = x
        best_acc = y
# print(best_acc, best_k)
acc_test = knn_model(best_k, train_feat, train_label, test_feat, test_label)
print("Accuracy of Testing Data", acc_test)

# K-means
k_means_model = KMeans(n_clusters=2)
k_means_model.fit(norm_feat_df)

first_cluster = feat_df.loc[k_means_model.labels_ == 0, :]
second_cluster = feat_df.loc[k_means_model.labels_ == 1, :]

# Visualize the two clusters using MaxHR and Age
plt.scatter(first_cluster.loc[:, 'MaxHR'], first_cluster.loc[:, 'MaxHR'], color='red')
plt.scatter(second_cluster.loc[:, 'Age'], second_cluster.loc[:, 'Age'], color='blue')
plt.show()



