from src.ml.PreProcessing.preprocessing import PreProcessing
from src.ml.Visualization.Visualization_Functions import Visualization
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib. pyplot as plt
import math
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
import sys
import io
from sklearn.metrics import accuracy_score
class KNearestNeighbors():
    def __init__(self,predicted_column,path,categorical_columns,sheet_name=None,train_test_split=True,supplied_test_set=None,percentage_split=0.2):
        self.predicted_column = predicted_column
        self.path = path
        self.categorical_columns = categorical_columns
        self.sheet_name = sheet_name
        self.train_test_split = train_test_split
        self.supplied_test_set = supplied_test_set
        self.percentage_split = percentage_split
    def __get_data(self):
        Preprocess = PreProcessing(self.path, self.sheet_name)
        Preprocess.set_predicted_column(self.predicted_column)
        self.label_names=Preprocess.get_label_names()
        Preprocess.dropping_operations()
        Preprocess.label_encoding()
        Preprocess.fill_missing_values(self.categorical_columns)
        X_train, X_test, y_train, y_test = Preprocess.train_split_test(supplied_test_set=self.supplied_test_set
                                                                       , percentage_split=self.percentage_split,
                                                                       train_test_splitt=self.train_test_split)
        self.count = Preprocess.number_of_records()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        return True

    def training(self, train_test_split=True):
        self.__get_data()
        best_score = 0
        best_k = 1
        sqrt_of_records = int(math.sqrt(self.count))
        for x in range(sqrt_of_records - 5, sqrt_of_records + 5):
            self.regr = KNeighborsClassifier(n_neighbors=x)
            self.regr.fit(self.X_train, self.y_train)
            self.y_pred_linear = self.regr.predict(self.X_test)
            if accuracy_score(self.y_test,self.y_pred_linear) > best_score:
                best_score = accuracy_score(self.y_test,self.y_pred_linear)
                best_k = x
        self.regr = KNeighborsClassifier(n_neighbors=best_k)
        self.regr.fit(self.X_train,self.y_train)
        self.y_pred = self.regr.predict_proba(self.X_test)
        self.y_pred_linear = self.regr.predict(self.X_test)
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        print('Cross-Track error: %.3f'
      % log_loss(self.y_test,self.y_pred))
        print("Accuracy Of Model", best_score)
        print("best k value is : ", best_k)
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        return output

    def predict(self,*X):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        X=np.asarray(X)
        X=[X]
        print(self.regr.predict(X))
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        return output

    def visualize(self):
        X_labels = np.arange(len(self.y_test))
        plt.scatter(X_labels[0:20], self.y_test[0:20], color='black')
        plt.scatter(X_labels[0:20], self.y_pred_linear[0:20], color='blue')
        plt.xticks((X_labels[0:20]))
        plt.yticks(self.y_test[0:20])
        plt.figure(figsize=(50, 50))
        plt.savefig("knn_compared_test_and_prediction.png")

    def visualize_classes(self):
        visualize=Visualization()
        x_pca=visualize.Dimension_Reduction_with_PCA(self.X_train)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.y_train)
        plt.xlabel('First principle component')
        plt.ylabel('Second principle component')
        plt.savefig("classes_knn.png")

    def classification_report(self):
          target_names = self.label_names.astype(str)
          return classification_report(self.y_test, self.y_pred_linear.round(), target_names=target_names)