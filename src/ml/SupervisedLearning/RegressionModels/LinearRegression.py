from src.ml.PreProcessing.preprocessing import PreProcessing
import matplotlib. pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
import io
import numpy as np
from functools import partial
from hyperopt import hp,fmin,tpe,Trials
from hyperopt import space_eval

class LinearRegressionModel():
    def __init__(self,predicted_column,path,categorical_columns,sheet_name=0,train_test_split=True,supplied_test_set=None,percentage_split=0.2):
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
    def training(self,args={"fit_intercept":True}):
          self.__get_data()
          self.regr = linear_model.LinearRegression(**args)
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
          X_test1=self.X_test.transpose()
          X_test1=X_test1[3]
          print(X_test1.size,self.y_test.size)
          plt.scatter(X_test1, self.y_test,  color='black')
          plt.plot(X_test1, self.y_pred, color='blue', linewidth=3)
          plt.xticks(())
          plt.yticks(())

          plt.show()
    def hyperopt_optimization(self):
        self.__get_data()
        def define_space():
          space = hp.choice('regressor',[
                        {
                        'model': linear_model.LinearRegression,
                        'param':
                          {    
                              'fit_intercept':hp.choice('fit_intercept', [False,True]),
                              'n_jobs':hp.choice('n_jobs', [-1]),
                              'normalize':hp.choice('normalize', [False,True]),
                          }
                        }])
          return space
        def optimize(args):
            fit_intercept = args['param']['fit_intercept']
            n_jobs = args['param']['n_jobs']
            normalize = args['param']['normalize']
            model=linear_model.LinearRegression(fit_intercept=fit_intercept,normalize=normalize ,
                                    n_jobs=n_jobs)
            model.fit(self.X_train,self.y_train)
            preds=model.predict(self.X_test)
            #preds=model.predict_proba(self.X_test)
            accuracy=metrics.r2_score(self.y_test,preds)
            return accuracy
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
        fit_intercept = self.best_parameters['param']['fit_intercept']
        n_jobs = self.best_parameters['param']['n_jobs']
        normalize = self.best_parameters['param']['normalize']
        args={"fit_intercept":fit_intercept,"normalize":normalize,"n_jobs":n_jobs}
        print(self.training(args))
        print(self.visualize())
		self.visualize()
