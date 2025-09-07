import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
from kmeans import KMeans



def load_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    return X, y, feature_names

def compute_nmi(true_labels, predicted_labels):
    return normalized_mutual_info_score(true_labels, predicted_labels)

def plot_clusters(X, y_pred, centroids, filename, plot_type='scatter'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    if plot_type == 'scatter':
        for cluster in set(y_pred):
            plt.scatter(
                X[y_pred == cluster, 0], 
                X[y_pred == cluster, 1], 
                label=f'Cluster {cluster}'
            )
        plt.scatter(
            centroids[:, 0], centroids[:, 1], 
            s=200, c='black', marker='X', label='Centroids'
        )
        plt.title('K-Means Clustering (Scatter)')
    
    elif plot_type == 'line':
        # Plot points connected by line within each cluster (for demonstration)
        for cluster in set(y_pred):
            plt.plot(
                X[y_pred == cluster, 0], 
                X[y_pred == cluster, 1], 
                marker='o', label=f'Cluster {cluster}'
            )
        plt.scatter(
            centroids[:, 0], centroids[:, 1], 
            s=200, c='black', marker='X', label='Centroids'
        )
        plt.title('K-Means Clustering (Line)')
    
    elif plot_type == 'centroid_only':
        # Just show centroids
        plt.scatter(
            centroids[:, 0], centroids[:, 1], 
            s=300, c='black', marker='X', label='Centroids'
        )
        plt.title('Centroids Only')
    
    else:
        raise ValueError(f'Unknown plot_type: {plot_type}')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig(filename)
    plt.close()

    def preprocess_data(X):
    # Handle missing values (simple strategy: fill with column mean)
    X = pd.DataFrame(X)
    X = X.fillna(X.mean())
    
    # Feature scaling (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def plot_elbow_silhouette(X, max_k=10, filename_elbow='../results/plots/elbow.png', filename_silhouette='../results/plots/silhouette.png'):
    inertia = []
    silhouette = []
    K = range(2, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k)
        y_pred = kmeans.fit(X)
        inertia.append(kmeans.inertia)
        silhouette.append(silhouette_score(X, y_pred))
    
    # Plot Elbow
    plt.figure()
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(filename_elbow)
    plt.close()
    
    # Plot Silhouette
    plt.figure()
    plt.plot(K, silhouette, 'go-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores For Different k')
    plt.savefig(filename_silhouette)
    plt.close()
    
    print(f"Elbow plot saved to {filename_elbow}")
    print(f"Silhouette plot saved to {filename_silhouette}")

def plot_interactive_clusters(X, y_pred, centroids, filename='../results/plots/interactive_clusters.html'):
    df = pd.DataFrame(X, columns=['PC1', 'PC2'])
    df['Cluster'] = y_pred
    
    fig = px.scatter(df, x='PC1', y='PC2', color='Cluster', title='K-Means Clustering Results')
    fig.write_html(filename)
    print(f"Interactive plot saved to {filename}")