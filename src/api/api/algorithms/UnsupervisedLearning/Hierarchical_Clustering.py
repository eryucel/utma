from api.algorithms.PreProcessing.preprocessing import PreProcessing
from api.algorithms.Visualization.Visualization_Functions import Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from scipy.spatial import distance_matrix


class Hierarchical_Clustering_Model():
    def __init__(self, path, categorical_columns, n_cluster, sheet_name=None):
        self.categorical_columns = categorical_columns
        self.path = path
        self.sheet_name = sheet_name
        self.n_cluster = n_cluster

    def __get_data(self):
        self.Preprocess = PreProcessing(self.path, self.sheet_name)
        dropped_colums = self.Preprocess.dropping_operations()
        self.column_names = np.asarray(self.Preprocess.get_column_names())
        self.column_names = self.column_names.astype(str)
        self.column_names = np.asarray(self.column_names).tolist()
        self.column_names.append("Clustering Result")
        self.column_names = np.asarray(self.column_names)
        self.changed_columns, self.columns_data = self.Preprocess.label_encoding()
        self.Preprocess.fill_missing_values(self.categorical_columns)
        data = self.Preprocess.get_data()
        data, _ = self.Preprocess.min_max_scaling(data)
        self.data = data

    def training(self):
        # Notice you start at 2 clusters for silhouette coefficient
        self.__get_data()
        dist_matrix = distance_matrix(self.data, self.data)
        self.cluster = AgglomerativeClustering(n_clusters=self.n_cluster, linkage='complete')
        self.cluster.fit(self.data)

    def visualize(self, task_id):
        self.training()
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.data)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.cluster.labels_, s=50, cmap='viridis')
        plt.savefig("files/hierarchical_" + str(task_id) + ".png", dpi=100)
        return "hierarchical_" + str(task_id) + ".png"

    def return_clustered_data(self, task_id):
        final_data = self.Preprocess.reverse_min_max_scaling()
        final_data = self.Preprocess.reverse_label_encoding(self.changed_columns, self.columns_data)
        final_data = np.insert(final_data, len(self.column_names) - 1, self.cluster.labels_, axis=1)
        final_data = pd.DataFrame(final_data, columns=self.column_names)
        final_data.sort_values(by=['Clustering Result'], ascending=False, inplace=True)
        final_data = final_data.reset_index(drop=True)
        final_data.to_csv("files/hierarchical_result_" + str(task_id) + ".csv")
        return {'final_data': pd.DataFrame.to_json(final_data), 'results': "hierarchical_result_" + str(task_id) + ".csv"}
