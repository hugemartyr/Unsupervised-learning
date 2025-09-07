from utils import load_iris_data, compute_nmi, plot_clusters
from pca import PCA
from kmeans import KMeans

def main():
    X, y_true, feature_names = load_iris_data()
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=3)
    y_pred = kmeans.fit(X_reduced)
    nmi = compute_nmi(y_true, y_pred)
    print(f'Normalized Mutual Information (NMI): {nmi:.4f}')
    plot_clusters(X_reduced, y_pred, kmeans.centroids, filename='../results/plots/kmeans_result.png')
    print("Clustering plot saved to results/plots/kmeans_result.png")

if __name__ == '__main__':
    main()