from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from config.settings import Settings

def find_optimal_k(data):
    wcss = []
    sil_scores = []
    k_range = range(Settings.K_MIN, min(Settings.K_MAX + 1, len(data)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=Settings.N_INIT, max_iter=Settings.MAX_ITER, random_state=Settings.RANDOM_STATE)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        if k > 1:
            sil_scores.append(silhouette_score(data, kmeans.labels_))
        else:
            sil_scores.append(0)
            
    best_k = k_range[np.argmax(sil_scores)] if sil_scores else 3
    return best_k, wcss, sil_scores, list(k_range)

def run_clustering(data, k):
    model = KMeans(n_clusters=k, n_init=Settings.N_INIT, max_iter=Settings.MAX_ITER, random_state=Settings.RANDOM_STATE)
    labels = model.fit_predict(data)
    return labels, model
