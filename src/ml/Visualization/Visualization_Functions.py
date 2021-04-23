from sklearn.decomposition import PCA
class Visualization():

    def Dimension_Reduction_with_PCA(self, X_train):
        pca = PCA(n_components=2)
        pca.fit(X_train)
        x_train_pca = pca.transform(X_train)
        return x_train_pca