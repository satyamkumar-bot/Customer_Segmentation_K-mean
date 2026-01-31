import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA

def plot_elbow(wcss, k_range):
    return px.line(x=list(k_range), y=wcss, markers=True, title='Elbow Method', labels={'x':'K', 'y':'Inertia'})

def plot_silhouette(sil_scores, k_range):
    return px.bar(x=list(k_range), y=sil_scores, title='Silhouette Scores', labels={'x':'K', 'y':'Score'})

def plot_clusters_2d(data, labels):
    pca = PCA(n_components=2)
    comps = pca.fit_transform(data)
    df_pca = pd.DataFrame(comps, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = labels.astype(str)
    return px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', title='2D Cluster Visualization')

def plot_clusters_3d(data, labels):
    if data.shape[1] < 3: return None
    pca = PCA(n_components=3)
    comps = pca.fit_transform(data)
    df_pca = pd.DataFrame(comps, columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca['Cluster'] = labels.astype(str)
    return px.scatter_3d(df_pca, x='PCA1', y='PCA2', z='PCA3', color='Cluster', title='3D Cluster Visualization')

def plot_cluster_dist(labels):
    counts = pd.Series(labels).value_counts().reset_index()
    counts.columns = ['Cluster', 'Count']
    counts['Cluster'] = counts['Cluster'].astype(str)
    return px.bar(counts, x='Cluster', y='Count', color='Cluster', title='Cluster Sizes')
