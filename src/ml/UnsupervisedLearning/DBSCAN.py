

from src.ml.PreProcessing.preprocessing import PreProcessing
from src.ml.Visualization.Visualization_Functions import Visualization
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
        self.dropped_columns,self.dropped_column_data,self.dropped_columns_locs = self.Preprocess.dropping_operations()
        self.changed_columns, self.columns_data = self.Preprocess.label_encoding()
        self.Preprocess.fill_missing_values(self.categorical_columns)
        data = self.Preprocess.get_data()
        data, _ = self.Preprocess.min_max_scaling(data)
        self.data = data

    def prepare_for_training(self, ):
        self.__get_data()
        min_samples = 2 * self.data.shape[1]
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        plt.plot(distances)
        plt.savefig("Distances_DBSCAN")
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        kl = KneeLocator(distances, distances, curve="convex", direction="increasing")
        print("Best probable epsilon value for this dataset is:", kl.elbow)
        dist_matrix = distance_matrix(self.data, self.data)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        return output, kl.elbow, min_samples

    def training(self):
        # Notice you start at 2 clusters for silhouette coefficient
        output, epsilon, min_samples = self.prepare_for_training()

        self.cluster = DBSCAN(eps=epsilon, min_samples=min_samples)
        self.cluster.fit(self.data)
        return output

    def visualize(self):
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.data)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.cluster.labels_, s=50, cmap='viridis')
        plt.savefig("DBSCAN_results.png")

    def return_clustered_data(self,):
        final_data = self.Preprocess.reverse_min_max_scaling()
        final_data = self.Preprocess.reverse_label_encoding(self.changed_columns, self.columns_data)
        final_data = self.Preprocess.reverse_dropping_operations(self.dropped_columns,
                                                                 self.dropped_column_data,
                                                                 self.dropped_columns_locs)
        final_data['Clustering Result'] = self.predicted_values
        final_data.sort_values(by=['Clustering Result'], ascending=False, inplace=True)
        final_data = final_data.reset_index(drop=True)
        final_data.to_excel("DBSCAN_output.xlsx")
        return final_data.head()