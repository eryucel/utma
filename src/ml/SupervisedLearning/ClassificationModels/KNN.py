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
from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt import space_eval

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

    def training(self):
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
    def optimized_training(self,args):
        self.__get_data()
        self.regr = KNeighborsClassifier(**args)
        self.regr.fit(self.X_train, self.y_train)
        self.y_pred = self.regr.predict_proba(self.X_test)
        self.y_pred_linear = self.regr.predict(self.X_test)
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        print('Cross-Track error: %.3f'
      % log_loss(self.y_test,self.y_pred))
        print("Accuracy Of Model", accuracy_score(self.y_test,self.y_pred_linear))
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        return output
    def hyperopt_optimization(self):
      self.__get_data()
      def define_space():
        space = hp.choice('classifier',[
                      {
                       'model': KNeighborsClassifier,
                       'param':
                         {
                            "n_neighbors":hp.quniform("n_neighbors",1,int(math.sqrt(self.count))*2,1),
                            "weights":hp.choice("weights",["uniform", "distance"]),
                            "algorithm":hp.choice("algorithm",["auto", "ball_tree", "kd_tree", "brute"]),
                            "leaf_size":hp.randint("leaf_size",100),
                            "p":hp.choice("p",[1,2]),
                         }
                      }])
        return space
      def optimize(args):
          n_neighbors = args['param']['n_neighbors']
          n_neighbors=int(n_neighbors)
          weights = args['param']['weights']
          algorithm = args['param']['algorithm']
          leaf_size = args['param']['leaf_size']+10
          p = args['param']['p']

          model=KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p)
          model.fit(self.X_train,self.y_train)
          #preds=model.predict(self.X_test)
          preds=model.predict_proba(self.X_test)
          #accuracy=metrics.accuracy_score(self.y_test,preds)
          return log_loss(self.y_test,preds)
      optimziation_function=partial(optimize)
      trials=Trials()
      space=define_space()
      result=fmin(
          fn=optimziation_function,
          space=space,
          algo=tpe.suggest,
          max_evals=100, #bu değer değerlendirilecek
          trials=trials
      )
      self.best_parameters=space_eval(space,result)
      return self.best_parameters
    def run_optimized_model(self):
      n_neighbors = self.best_parameters['param']['n_neighbors']
      n_neighbors=int(n_neighbors)
      weights = self.best_parameters['param']['weights']
      algorithm = self.best_parameters['param']['algorithm']
      leaf_size = self.best_parameters['param']['leaf_size']+10
      p = self.best_parameters['param']['p']
      args={"n_neighbors":n_neighbors,"weights":weights,"algorithm":algorithm,"leaf_size":leaf_size,"p":p}
      print(self.optimized_training(args))
      print(self.classification_report())
      print(self.visualize())