from src.ml.PreProcessing.preprocessing import PreProcessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import io
import sys
from functools import partial
from hyperopt import hp,fmin,tpe,Trials
from hyperopt import space_eval
class LassoRegressionModel():
    def __init__(self, predicted_column, path, categorical_columns,sheet_name=0,train_test_split=True,supplied_test_set=None,percentage_split=0.2):
        self.predicted_column = predicted_column
        self.path = path
        self.categorical_columns = categorical_columns
        self.sheet_name = sheet_name
        self.train_test_split = train_test_split
        self.supplied_test_set = supplied_test_set
        self.percentage_split = percentage_split

    def __get_data(self, train_test_split=True):
        Preprocess = PreProcessing(self.path, self.sheet_name)
        Preprocess.set_predicted_column(self.predicted_column)
        Preprocess.dropping_operations()
        Preprocess.label_encoding()
        Preprocess.fill_missing_values(self.categorical_columns)
        X_train, X_test, y_train, y_test = Preprocess.train_split_test(supplied_test_set=self.supplied_test_set
                                                                       , percentage_split=self.percentage_split,
                                                                       train_test_splitt=self.train_test_split)
        X_train, X_test = Preprocess.min_max_scaling(X_train, X_test)
        # X_train, X_test=Preprocess.GaussianTranformation(X_train, X_test)
        # X_train, X_test=Preprocess.Normalization(X_train, X_test)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
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

    def training(self,args={"alpha":0.1} ):
        self.__get_data()
        self.regr = linear_model.Lasso(**args)
        self.regr.fit(self.X_train, self.y_train)
        self.y_pred = self.regr.predict(self.X_test)
        print('Mean squared error: %.2f'
              % mean_squared_error(self.y_pred, self.y_test))
        return self.score_estimator(self.y_pred, self.y_test, self.predicted_column)

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
        # print(X_test1.size,self.y_test.size)
        plt.scatter(X_labels[0:20], self.y_test[0:20], color='black')
        plt.scatter(X_labels[0:20], self.y_pred[0:20], color='blue')
        plt.xticks((X_labels[0:20]))
        plt.yticks(self.y_test[0:20])
        plt.figure(figsize=(100, 100))
        plt.savefig("Lasso_compared_test_and_prediction.png")


    def hyperopt_optimization(self):
      self.__get_data()
      def define_space():
        space = hp.choice('regressor',[
                      {
                       'model': linear_model.Lasso,
                       'param':
                         {    
                            'alpha':hp.uniform('alpha',0.1,1.0 ), #np.arange(0, 1, 0.01)
                            'fit_intercept':hp.choice('fit_intercept', [True,False]),
                            'normalize':hp.choice('normalize', [True,False]),
                            'random_state':hp.choice('random_state',range(9, 51, 3)),
                            'max_iter':hp.choice('max_iter', range(1000, 3000, 100))
                         }
                      }])
        return space
      def optimize(args):
          alpha = args['param']['alpha']
          fit_intercept = args['param']['fit_intercept']
          normalize = args['param']['normalize']
          random_state = args['param']['random_state']
          max_iter = args['param']['max_iter']
          model=linear_model.Lasso(alpha = alpha , fit_intercept=fit_intercept,normalize=normalize ,
                                   random_state=random_state,max_iter=max_iter)
          model.fit(self.X_train,self.y_train)
          preds=model.predict(self.X_test)
          #preds=model.predict_proba(self.X_test)
          accuracy=metrics.r2_score(self.y_test,preds)
          return -1.0*accuracy
      import warnings
      warnings.filterwarnings('ignore')
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
        alpha = self.best_parameters['param']['alpha']
        fit_intercept = self.best_parameters['param']['fit_intercept']
        normalize = self.best_parameters['param']['normalize']
        random_state = self.best_parameters['param']['random_state']
        max_iter = self.best_parameters['param']['max_iter']
        args={"alpha":alpha,"fit_intercept":fit_intercept,"normalize":normalize,"random_state":random_state,"max_iter":max_iter}
        print(self.training(args))
        print(self.visualize())
		self.visualize()