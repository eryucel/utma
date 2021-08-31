from api.algorithms.PreProcessing.preprocessing import PreProcessing
from api.algorithms.Visualization.Visualization_Functions import Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import sys
import io
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
from kneed import KneeLocator


class DBSCAN_Model():
    def __init__(self, path, categorical_columns, sheet_name=None):
        self.categorical_columns = categorical_columns
        self.path = path
        self.sheet_name = sheet_name

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

    def prepare_for_training(self, task_id):
        self.__get_data()
        min_samples = 2 * self.data.shape[1]
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        plt.plot(distances)
        plt.savefig("files/DBSCAN_distances_" + str(task_id) + '.png')
        # old_stdout = sys.stdout
        # new_stdout = io.StringIO()
        # sys.stdout = new_stdout
        kl = KneeLocator(distances, distances, curve="convex", direction="increasing")
        # print("Best probable epsilon value for this dataset is:", kl.elbow)
        dist_matrix = distance_matrix(self.data, self.data)
        # output = new_stdout.getvalue()
        # sys.stdout = old_stdout
        return kl.elbow, min_samples

    def training(self, task_id):
        # Notice you start at 2 clusters for silhouette coefficient
        epsilon, min_samples = self.prepare_for_training(task_id)

        self.cluster = DBSCAN(eps=epsilon, min_samples=min_samples)
        self.cluster.fit(self.data)
        return {'probable_epsilon': epsilon, 'distances': 'DBSCAN_distances_' + str(task_id) + '.png'}

    def visualize(self, task_id):
        self.training(task_id)
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.data)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.cluster.labels_, s=50, cmap='viridis')
        plt.savefig("files/DBSCAN_" + str(task_id) + ".png", dpi=100)
        return "DBSCAN_" + str(task_id) + ".png"

    def return_clustered_data(self, task_id):
        final_data = self.Preprocess.reverse_min_max_scaling()
        final_data = self.Preprocess.reverse_label_encoding(self.changed_columns, self.columns_data)
        final_data = np.insert(final_data, len(self.column_names) - 1, self.cluster.labels_, axis=1)
        final_data = pd.DataFrame(final_data, columns=self.column_names)
        final_data.sort_values(by=['Clustering Result'], ascending=False, inplace=True)
        final_data = final_data.reset_index(drop=True)
        final_data.to_csv("files/DBSCAN_result_" + str(task_id) + ".csv")
        return {'final_data': pd.DataFrame.to_json(final_data), 'results': "DBSCAN_result_" + str(task_id) + ".csv"}
