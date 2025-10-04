import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import gensim.downloader as api
from sklearn.cluster import KMeans
import multiprocessing

# ========= 参数 =========
win_size = [3, 7, 13, 25]
vector_size = [20, 70, 100, 300]
cluster_words = ['yen', 'yuan', 'spain', 'brazil', 'africa', 'asia']

expected_groups = [
    {'yen', 'yuan'},
    {'spain', 'brazil'},
    {'africa', 'asia'}
]

# ========= 工具函数 =========
def score_clustering(labels, words):
    """评估聚类结果是否把预期的group聚到一起"""
    wl = {w: l for w, l in zip(words, labels)}
    score = 0
    for group in expected_groups:
        present = [w for w in group if w in wl]
        if len(present) >= 2:
            labs = {wl[w] for w in present}
            if len(labs) == 1:  # 同组的词在同一类
                score += 1
    return score

def rank_of_word(sim_list, target="son"):
    """返回目标词在similar_by_vector中的排名"""
    for i, (w, _) in enumerate(sim_list, start=1):
        if w == target:
            return i
    return 9999  # 没找到

def train_and_evaluate(corpus, window, vector):
    """训练模型并评估 analogy + clustering"""
    model = Word2Vec(
        sentences=corpus,
        window=window,
        vector_size=vector,
        sg=1,                 # skip-gram
        epochs=2,
        min_count=10,
        workers=multiprocessing.cpu_count()
    )

    # ---- 1) Analogy: man - woman + daughter ≈ son ----
    analogy_rank, analogy_top = 9999, []
    if all(w in model.wv for w in ["man", "woman", "daughter"]):
        transform = model.wv["man"] - model.wv["woman"]
        target_vec = transform + model.wv["daughter"]
        analogy_top = model.wv.similar_by_vector(target_vec, topn=10)
        analogy_rank = rank_of_word(analogy_top, target="son")

    # ---- 2) Clustering ----
    avail_words = [w for w in cluster_words if w in model.wv]
    cluster_score, cluster_labels = -1, []
    if len(avail_words) >= 2:
        X = np.vstack([model.wv[w] for w in avail_words])
        n_clusters = min(3, len(avail_words))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        km.fit(X)
        cluster_labels = list(km.labels_)
        cluster_score = score_clustering(cluster_labels, avail_words)

    return {
        "window": window,
        "vector_size": vector,
        "analogy_rank_son": analogy_rank,
        "analogy_top10": analogy_top[:5],
        "cluster_words": avail_words,
        "cluster_labels": cluster_labels,
        "cluster_score": cluster_score
    }

# ========= 主程序 =========
def main():
    corpus = list(api.load('text8'))
    results = []

    for window in win_size:
        for vector in vector_size:
            res = train_and_evaluate(corpus, window, vector)
            results.append(res)

    df = pd.DataFrame(results)
    # 按 知识类比效果优先，其次聚类得分 排序
    best = df.sort_values(by=["analogy_rank_son", "cluster_score"], ascending=[True, False]).iloc[0]

    print("\n===== Best Hyper-Parameters =====")
    print(f"window={best['window']}, vector_size={best['vector_size']}")
    print(f"Analogy: 'son' rank={best['analogy_rank_son']}")
    print(f"Cluster score={best['cluster_score']}")
    print("Analogy top10:", best['analogy_top10'])
    print("Clustering:", list(zip(best['cluster_words'], best['cluster_labels'])))

    return df

if __name__ == "__main__":
    df_results = main()
