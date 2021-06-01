from api.algorithms.PreProcessing.preprocessing import PreProcessing
from api.algorithms.Visualization.Visualization_Functions import Visualization
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
        self.training()

    def __get_data(self):
        self.Preprocess = PreProcessing(self.path, self.sheet_name)
        dropped_colums = self.Preprocess.dropping_operations()
        self.column_names = np.asarray(self.Preprocess.get_column_names())
        self.column_names = self.column_names.astype(str)
        self.column_names = np.asarray(self.column_names).tolist()
        self.column_names.append("Clustering Result")
        self.column_names = np.asarray(self.column_names)
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

    def training(self):
        # Notice you start at 2 clusters for silhouette coefficient
        self.__get_data()
        for k in range(1, 11):
            self.cluster = KMeans(n_clusters=k, **self.kmeans_kwargs)
            self.cluster.fit(self.data)
            self.sse.append(self.cluster.inertia_)

    def best_k_value(self):
        kl = KneeLocator(range(1, 11), self.sse, curve="convex", direction="decreasing")
        self.best_value = kl.elbow
        return kl.elbow.astype(str)

    def finalize_model(self):
        self.cluster = KMeans(n_clusters=self.best_value, **self.kmeans_kwargs)
        self.cluster.fit(self.data)

    def predict(self):
        self.predicted_values = self.cluster.predict(self.data)
        return self.predicted_values

    def visualize(self, task_id):
        plt.style.use("fivethirtyeight")
        plt.plot(range(1, 11), self.sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.savefig("files/kMeans_best_n_value_" + str(task_id) + ".png", dpi=100)
        return "kMeans_best_n_value_" + str(task_id) + ".png"

    def visualize1(self, task_id):
        self.finalize_model()
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.data)
        predicted_values = self.predict()
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=predicted_values, s=50, cmap='viridis')
        plt.savefig("files/kMeans_" + str(task_id) + ".png", dpi=100)
        return "kMeans_" + str(task_id) + ".png"

    def return_clustered_data(self, task_id):
        final_data = self.Preprocess.reverse_min_max_scaling()
        final_data = self.Preprocess.reverse_label_encoding(self.changed_columns, self.columns_data)
        final_data = np.insert(final_data, len(self.column_names) - 1, self.predicted_values, axis=1)
        final_data = pd.DataFrame(final_data, columns=self.column_names)
        final_data.sort_values(by=['Clustering Result'], ascending=False, inplace=True)
        final_data = final_data.reset_index(drop=True)
        final_data.to_csv("files/kMeans_result_" + str(task_id) + ".csv")
        return {'final_data': pd.DataFrame.to_json(final_data), 'results': "kMeans_result_" + str(task_id) + ".csv"}
