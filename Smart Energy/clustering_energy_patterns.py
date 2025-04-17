def cluster_days(df):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(df_pca)

    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.title("Clustering of Daily Energy Usage Patterns")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()
