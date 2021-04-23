from preprocessing import PreProcessing
import matplotlib. pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
import io
import numpy as np
class AdaBoost_Regressor():
  def __init__(self,predicted_column,path,categorical_columns,sheet_name=True,train_test_split=True,supplied_test_set=None,percentage_split=0.2):
      self.predicted_column = predicted_column
      self.categorical_columns=categorical_columns
      self.path = path
      self.sheet_name = sheet_name
      self.train_test_split = train_test_split
      self.supplied_test_set = supplied_test_set
      self.percentage_split = percentage_split
  def __get_data(self,train_test_split=True):
      Preprocess=PreProcessing(self.path)
      Preprocess.set_predicted_column(self.predicted_column)
      Preprocess.dropping_operations()
      Preprocess.label_encoding()
      Preprocess.fill_missing_values(self.categorical_columns)
      X_train, X_test, y_train, y_test = Preprocess.train_split_test(supplied_test_set=self.supplied_test_set
                                                                     , percentage_split=self.percentage_split,
                                                                     train_test_splitt=self.train_test_split)
      self.X_train=X_train
      self.X_test=X_test
      self.y_train=y_train
      self.y_test=y_test
      return True

  def score_estimator(self, y_pred, test_data, predicted_column):
      """Score an estimator on the test set."""
      old_stdout = sys.stdout
      new_stdout = io.StringIO()
      sys.stdout = new_stdout
      print("MSE: %.3f" %
            mean_squared_error(test_data, y_pred))
      print("MAE: %.3f" %
            mean_absolute_error(test_data, y_pred))
      print("Accuracy Of Model", metrics.r2_score( self.y_test,self.y_pred))
      output = new_stdout.getvalue()
      sys.stdout = old_stdout
      return output
  def training(self):
        self.__get_data()
        self.regr = AdaBoostRegressor(random_state=0, n_estimators=100)
        self.regr.fit(self.X_train, self.y_train)
        self.y_pred = self.regr.predict(self.X_test)
        return self.score_estimator(self.y_pred, self.y_test, self.predicted_column)

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
        X_labels = np.arange(len(self.y_test))
        # print(X_test1.size,self.y_test.size)
        plt.scatter(X_labels[0:20], self.y_test[0:20], color='black')
        plt.scatter(X_labels[0:20], self.y_pred[0:20], color='blue')
        plt.xticks((X_labels[0:20]))
        plt.yticks(self.y_test[0:20])
        plt.figure(figsize=(100, 100))
        plt.savefig("SGD_compared_test_and_prediction.png")
