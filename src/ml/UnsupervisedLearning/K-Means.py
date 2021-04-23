from python.ml.PreProcessing.preprocessing import PreProcessing
from python.ml.Visualization.Visualization_Functions import Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import io
from sklearn.cluster import KMeans
from kneed import KneeLocator


class K_MeansModel():
    def __init__(self, path, categorical_columns, sheet_name=None, ):
        self.categorical_columns = categorical_columns
        self.path = path
        self.sheet_name = sheet_name
        self.sse = []
        self.best_value = 0

    def __get_data(self):
        self.Preprocess = PreProcessing(self.path, self.sheet_name)
        self.dropped_columns,self.dropped_column_data,self.dropped_columns_locs = self.Preprocess.dropping_operations()
        self.changed_columns, self.columns_data = self.Preprocess.label_encoding()
        # print(self.changed_columns)
        self.Preprocess.fill_missing_values(self.categorical_columns)
        data = self.Preprocess.get_data()
        # print(data[0:5])
        data, _ = self.Preprocess.min_max_scaling(data)
        # print(data[0:5])
        self.data = data
        self.kmeans_kwargs = kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42, }
        return True

    def training(self, train_test_split=True, ):
        # Notice you start at 2 clusters for silhouette coefficient
        self.__get_data()
        for k in range(1, 11):
            self.cluster = KMeans(n_clusters=k, **self.kmeans_kwargs)
            self.cluster.fit(self.data)
            self.sse.append(self.cluster.inertia_)

    def best_k_value(self):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        kl = KneeLocator(range(1, 11), self.sse, curve="convex", direction="decreasing")
        print("Best probable cluster number for this dataset is:", kl.elbow)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        self.best_value = kl.elbow
        return output

    def finalize_model(self):
        self.cluster = KMeans(n_clusters=self.best_value, **self.kmeans_kwargs)
        self.cluster.fit(self.data)

    def predict(self):
        self.predicted_values = self.cluster.predict(self.data)
        return self.predicted_values

    def visualize(self):
        plt.style.use("fivethirtyeight")
        plt.plot(range(1, 11), self.sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.savefig("best_n_value_k-means.png")

    def visualize1(self):
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.data)
        predicted_values = self.predict()
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=predicted_values, s=50, cmap='viridis')
        plt.savefig("clustering_results.png")

    def return_clustered_data(self):
        final_data = self.Preprocess.reverse_min_max_scaling()
        final_data = self.Preprocess.reverse_label_encoding(self.changed_columns, self.columns_data)
        final_data = self.Preprocess.reverse_dropping_operations(self.dropped_columns,
                                                                 self.dropped_column_data,
                                                                 self.dropped_columns_locs)
        final_data['Clustering Result'] = self.predicted_values
        final_data.sort_values(by=['Clustering Result'], ascending=False, inplace=True)
        final_data = final_data.reset_index(drop=True)
        final_data.to_excel("kmeans_output.xlsx")
        return final_data