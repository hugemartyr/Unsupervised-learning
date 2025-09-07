import streamlit as st
import pandas as pd
from utils import preprocess_data, plot_elbow_silhouette, plot_interactive_clusters
from pca import PCA
from kmeans import KMeans
from sklearn.datasets import load_iris

def main():
    st.title("Unsupervised Clustering Dashboard")

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", data.head())

        X = data.values
        X = preprocess_data(X)

        if st.button("Run Elbow & Silhouette Analysis"):
            plot_elbow_silhouette(X)
            st.image('../results/plots/elbow.png', caption='Elbow Plot')
            st.image('../results/plots/silhouette.png', caption='Silhouette Plot')

        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
        if st.button("Run Clustering"):
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)

            kmeans = KMeans(n_clusters=n_clusters)
            y_pred = kmeans.fit(X_reduced)

            plot_interactive_clusters(X_reduced, y_pred, kmeans.centroids)
            st.write("Interactive plot saved. [Open it](../results/plots/interactive_clusters.html)")

            nmi = 0  # No ground truth label in generic dataset
            st.write(f"Clustering done with {n_clusters} clusters. NMI not calculated.")

if __name__ == "__main__":
    main()
