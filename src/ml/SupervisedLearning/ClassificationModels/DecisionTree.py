from src.ml.PreProcessing.preprocessing import PreProcessing
from src.ml.Visualization.Visualization_Functions import Visualization
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from six import StringIO
import matplotlib.image as mpimg
from sklearn import tree
import pydotplus
import sys
import io
import numpy as np
class DecisionTree():
  def __init__(self,predicted_column,path,categorical_columns,sheet_name=None,train_test_split=True,supplied_test_set=None,percentage_split=0.2):
        self.predicted_column=predicted_column
        self.path=path
        self.categorical_columns = categorical_columns
        self.sheet_name = sheet_name
        self.train_test_split=train_test_split
        self.supplied_test_set=supplied_test_set
        self.percentage_split=percentage_split

  def __get_data(self):
        Preprocess=PreProcessing(self.path, self.sheet_name)
        self.columns=Preprocess.get_column_names()
        self.columns=np.delete(self.columns,self.predicted_column)
        Preprocess.set_predicted_column(self.predicted_column)
        self.label_names=Preprocess.get_label_names()
        Preprocess.dropping_operations()
        Preprocess.label_encoding()
        Preprocess.fill_missing_values(self.categorical_columns)
        X_train, X_test, y_train, y_test=Preprocess.train_split_test(supplied_test_set=self.supplied_test_set
                                                                     ,percentage_split=self.percentage_split,
                                                                     train_test_splitt=self.train_test_split)
        self.X_train=X_train
        #self.X_test=X_test.astype(int)
        self.X_test=X_test
        self.y_train=y_train
        #self.y_test=y_test.astype(int)
        self.y_test=y_test
        return True

  def training(self,train_test_split=True):
        self.__get_data()
        self.regr =DecisionTreeClassifier(criterion="entropy", max_depth = 10)
        self.regr.fit(self.X_train, self.y_train)
        self.y_pred = self.regr.predict_proba(self.X_test)
        self.y_pred_tree = self.regr.predict(self.X_test)
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        print('Cross-Track error: %.3f'
          % log_loss(self.y_test,self.y_pred))
        #print("Accuracy Of Model",self.regr.score(self.X_test, self.y_test))
        print("DecisionTrees's Accuracy: ", metrics.accuracy_score(self.y_test, self.y_pred_tree))
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        return output

  def predict(self, *X):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        X = np.asarray(X)
        X = [X]
        print(self.regr.predict(X))
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        return output

  def visualize(self):
        dot_data = StringIO()
        filename = "drugtree.png"
        featureNames = self.columns
        targetNames = self.label_names
        out=tree.export_graphviz(self.regr,feature_names=featureNames, out_file=dot_data, class_names= self.label_names, filled=True,  special_characters=True,rotate=False)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(filename)
        img = mpimg.imread(filename)
        plt.figure(figsize=(100, 200))
        plt.imshow(img,interpolation='nearest')
  def visualize_classes(self):
        visualize=Visualization()
        x_pca=visualize.Dimension_Reduction_with_PCA(self.X_train)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.y_train)
        plt.xlabel('First principle component')
        plt.ylabel('Second principle component')
        plt.savefig("decisiontree_classes.png")

  def classification_report(self):
        target_names = self.label_names.astype(str)
        return classification_report(self.y_test, self.y_pred_tree.round(), target_names=target_names)