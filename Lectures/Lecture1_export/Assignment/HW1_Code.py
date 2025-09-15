import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# -Split data into training (72%), validation (8%) and test (20%)
# -Train three models of KNN-3, KNN-9 and KNN-21 using training data to predict label (last column) based on the other columns (features)
# -Calculate the accuracy of three models on the validation data
# -Select the best K among these three models
# -Train a model using the best K on both training and validation data
# -Calculate the accuracy of the model on test data
# -Train a k-means model with two clusters on all feature data
# -Visualize the two clusters using MaxHR and Age

org_df = pd.read_csv('Heart_Failure.csv')

# Clean the outlier
for column in org_df:
    q3, q1 = np.percentile(org_df[column], [75,25])
    fence = (q3 - q1) * 1.5
    upper = q3 + fence
    lower = q1 - fence
    org_df.loc[(org_df[column] > upper) | (org_df[column] < lower), column] = np.nan

# Impute missing values
df_encoded = pd.get_dummies(org_df, drop_first=True)
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_dataset = imputer.fit_transform(df_encoded)
org_df = pd.DataFrame(imputed_dataset, columns=df_encoded.columns)

# Define the features and label for KNN
label_df = org_df.loc[:, org_df.columns == 'HeartDisease']
feat_df = org_df.loc[:, org_df.columns != 'HeartDisease']

# #Encoding Categorical Variables
# feat_df = pd.get_dummies(feat_df, dtype='int')

#Normalize Data
norm_feat_df = (feat_df - feat_df.mean()) / feat_df.std()

# Split data into training (72%), validation (8%) and test (20%)
t_feat, test_feat, t_label, test_label = train_test_split(norm_feat_df,label_df,test_size=0.2)
train_feat, val_feat, train_label, val_label = train_test_split(t_feat,t_label,test_size=0.1)

# KNN
def knn_model(k, train_feat, train_label, val_feat, val_label):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_feat, train_label)
    val_y = knn.predict(val_feat)
    # val_x = knn.predict(train_feat)
    acc = accuracy_score(val_y, val_label)
    # acc1 = accuracy_score(val_x, train_label)
    return acc

KNN_model = [3, 9, 21]
acc_val = []
# acc_train = []
for k in KNN_model:
    acc = knn_model(k, train_feat, train_label, val_feat, val_label)
    acc_val.append(acc)
    # acc_train.append(acc1)
# # Graph
# plt.plot(KNN_model, acc_val, marker='o', label='Validation Accuracy')
# plt.plot(KNN_model, acc_train, marker='s', label='Training Accuracy')
# plt.xlabel('k (Number of Neighbors)')
# plt.ylabel('Accuracy')
# plt.title('KNN vs k')
# plt.legend()
# plt.grid(True)
# plt.show()

# KNN result for the best model
best_k = 0
best_acc = 0
for x, y in zip(KNN_model, acc_val):
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



