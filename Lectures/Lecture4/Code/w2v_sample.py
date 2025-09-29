from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
from sklearn.cluster import KMeans
import numpy as np

# Training data
corpus = api.load('text8')

# Train Word2Vec model
model = Word2Vec(sentences=corpus, window=13, vector_size=300)

# Print embedding vector of Paris
print("Embedding vector for Paris is: ", model.wv['paris'])

# Find most similar embeddings to France
print('Similar to France: ', model.wv.similar_by_vector (model.wv['france'],topn=3))

# Find most similar embeddings to a transformed embedding
transform = model.wv['france'] - model.wv['paris']
print('Transform: ', model.wv.similar_by_vector ( transform + model.wv['madrid'] ,topn=3))

# Some word embeddings
embeddings =np.array([
model.wv['paris'] , model.wv['he'],
model.wv['vienna'] , model.wv['she']
])

# K-means clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(embeddings)

# Print cluster assignments
for i, label in enumerate(kmeans.labels_):
    print("Embedding ", i, " is in cluster ", label)
