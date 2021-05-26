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

class SGDRegressionModel():
    def __init__(self, predicted_column, path, categorical_columns,sheet_name=0,train_test_split=True,supplied_test_set=None,percentage_split=0.2):
        self.predicted_column = predicted_column
        self.categorical_columns=categorical_columns
        self.path = path
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

    def training(self,args={"max_iter":1000, "tol":1e-3}, train_test_split=True, ):
        self.__get_data()
        self.regr = linear_model.SGDRegressor(**args)
        self.regr.fit(self.X_train, self.y_train)
        self.y_pred = self.regr.predict(self.X_test)
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
        plt.savefig("SGD_compared_test_and_prediction.png")

    def hyperopt_optimization(self):
        self.__get_data()
        def define_space():
          space = hp.choice('regressor',[
                        {
                        'model': linear_model.SGDRegressor,
                        'param':
                          {    
                              'loss_group' :hp.choice('loss_group',
                                             [
                                                {
                                                 'loss':hp.choice('loss1', ['squared_loss']),
                                                 'epsilon':hp.choice('epsilon1',[0.1])
                                                },
                                                {
                                                 'loss':hp.choice('loss2', ['huber','epsilon_insensitive', 'squared_epsilon_insensitive']),
                                                 'epsilon':hp.choice('epsilon2',np.arange(0, 1, 0.1)),
                                                },
                                             ]),
                              'penalty':hp.choice('penalty',['l2','l1','elasticnet'] ),
                              'alpha':hp.choice('alpha',np.arange(0.01, 1, 0.01)),
                              'fit_intercept':hp.choice('fit_intercept', [False]),
                              'random_state':hp.choice('random_state', [range(9, 51, 3)]),
                              'max_iter':hp.choice('max_iter', range(1000, 3000, 100)),
                              'learning_rate':hp.choice('learning_rate', ['constant','optimal','invscaling','adaptive']), 
                              'tol':hp.choice('tol', [1e-3]),
                          }
                        }])
          return space
        def optimize(args):
            penalty = args['param']['penalty']
            # l1_ratio = args['param']['hyper_param_groups']['l1_ratio']
            loss = args['param']['loss_group']['loss']
            epsilon = args['param']['loss_group']['epsilon']
            fit_intercept = args['param']['fit_intercept']
            alpha = args['param']['alpha']
            max_iter = args['param']['max_iter']
            learning_rate = args['param']['learning_rate']
            tol = args['param']['tol']
            
            model=linear_model.SGDRegressor(fit_intercept=fit_intercept,penalty=penalty ,
                                                loss=loss , epsilon=epsilon,alpha=alpha , 
                                                max_iter=max_iter,learning_rate=learning_rate ,tol=tol)
            model.fit(self.X_train,self.y_train)
            preds=model.predict(self.X_test)
            #preds=model.predict_proba(self.X_test)
            accuracy=metrics.r2_score(self.y_test,preds)
            return -1*accuracy # chek <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
        penalty = self.best_parameters['param']['penalty']
        # l1_ratio = self.best_parameters['param']['hyper_param_groups']['l1_ratio']
        loss = self.best_parameters['param']['loss_group']['loss']
        epsilon = self.best_parameters['param']['loss_group']['epsilon']
        fit_intercept = self.best_parameters['param']['fit_intercept']
        alpha = self.best_parameters['param']['alpha']
        max_iter = self.best_parameters['param']['max_iter']
        learning_rate = self.best_parameters['param']['learning_rate']
        tol = self.best_parameters['param']['tol']
        args={"fit_intercept":fit_intercept,"penalty":penalty ,
              "loss":loss ,"epsilon":epsilon,"alpha":alpha ,
              "max_iter":max_iter,"learning_rate":learning_rate ,"tol":tol}

        print(self.training(args))
        print(self.visualize())