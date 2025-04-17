def run_pca(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    pca = PCA()
    components = pca.fit_transform(df_scaled)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Variance Explained')
    plt.grid(True)
    plt.show()
