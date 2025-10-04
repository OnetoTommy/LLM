import numpy as np
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
from sklearn.cluster import KMeans

# Train Word2Vec models on text8 data using all combinations of the following hyper-parameters (16 models):
# win_size = [3,7,13,25]
# vector_size = [20,70,100,300]
# For each model, perform the following tasks:
# Define Transform as Embedding('man') - Embedding( 'woman')
# Use similar_by_vector method to find the most similar embedding to (Transform + Embedding('daughter’))
# Cluster the following embeddings created by the model using K-means (K=3):
#    'yen', 'yuan', 'spain', 'brazil', 'africa', 'asia’
# By assessment of the results of transformation and clustering, choose the best set of hyper-parameters to
# capture relationships between the concepts.

# training corpus data
corpus = list(api.load('text8'))
# print(corpus)

#Hyper-parameters -- win_size and vector_size
win_size = [3,7,13,25]
vector_size = [20,70,100,300]

# Define the array of cluster
cluster_words = ['yen', 'yuan', 'spain', 'brazil', 'africa', 'asia']

# Expected Group
expected_groups = [
    {'yen', 'yuan'},
    {'spain', 'brazil'},
    {'africa', 'asia'}
]

# Cluster Function
def score_clustering(labels, words):
    wl = {w: l for w, l in zip(words, labels)}
    score = 0
    for group in expected_groups:
        present = [w for w in group if w in wl]
        if len(present) >= 2:
            labs = {wl[w] for w in present}
            if len(labs) == 1:
                score += 1
    return score

# Define the rank of word
def rank_of_word(sim_list, target="son"):
    for i, (w, _) in enumerate(sim_list, start=1):
        if w == target:
            return i
    return 9999

# Array of Answer
result = []

# Build models based on different Hyper-parameters
# for i in range(2):
for window in win_size:
    for vector in vector_size:
        model = Word2Vec(sentences=corpus, window=window, vector_size=vector,
                         sg=1, epochs=2, min_count=10)
        if all(word in model.wv for word in ['man', 'woman', 'daughter']):
            transform = model.wv['man'] - model.wv['woman']
            target_vec = transform + model.wv["daughter"]
            res = model.wv.similar_by_vector(target_vec, topn=3)

        analogy_rank, analogy_top = 9999, []
        if all(w in model.wv for w in ["man", "woman", "daughter"]):
            transform = model.wv["man"] - model.wv["woman"]
            target_vec = transform + model.wv["daughter"]
            analogy_top = model.wv.similar_by_vector(target_vec, topn=3)
            analogy_rank = rank_of_word(analogy_top, target="son")

        #Cluster words
        avail_words = [w for w in cluster_words if w in model.wv]
        cluster_score, cluster_labels = -1, []
        if len(avail_words) >= 3:
            X = np.vstack([model.wv[w] for w in avail_words])
            km = KMeans(n_clusters=3, n_init=10, random_state=42)
            km.fit(X)
            cluster_labels = list(km.labels_)
            cluster_score = score_clustering(cluster_labels, avail_words)

        # Append the result
        result.append({
            "window": window,
            "vector_size": vector,
            "analogy_rank_son": analogy_rank,
            "analogy_top10": analogy_top,
            "cluster_words": avail_words,
            "cluster_labels": cluster_labels,
            "cluster_score": cluster_score
        })

# Select the Best result
best = min(result, key=lambda r: (r["analogy_rank_son"], -r["cluster_score"]))

print("\n===== Best Hyper-Parameters =====")
print(f"window={best['window']}, vector_size={best['vector_size']}")
print(f"Analogy: 'son' rank={best['analogy_rank_son']}")
print(f"Cluster score={best['cluster_score']}")
print("Analogy top10:", best['analogy_top10'])
print("Clustering:", list(zip(best['cluster_words'], best['cluster_labels'])))
