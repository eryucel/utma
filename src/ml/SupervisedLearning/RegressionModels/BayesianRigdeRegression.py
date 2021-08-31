from api.algorithms.PreProcessing.preprocessing import PreProcessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import numpy as np
import io
import sys
from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt import space_eval


class BayesianRigdeRegressionModel():
    def __init__(self, predicted_column, path, categorical_columns, sheet_name=0, train_test_split=True,
                 supplied_test_set=None, percentage_split=0.2):
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
        result = {'MSE': mean_squared_error(test_data, y_pred),
                  'MAE': mean_absolute_error(test_data, y_pred),
                  'accuracy': metrics.r2_score(self.y_test, self.y_pred)}
        return result

    def training(self, args={}):
        self.__get_data()
        self.regr = linear_model.BayesianRidge(**args)
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

    def visualize(self, task_id, optimized=False):
        plt.clf()
        X_labels = np.arange(len(self.y_test))
        # print(X_test1.size,self.y_test.size)
        plt.scatter(X_labels[0:20], self.y_test[0:20], color='black')
        plt.scatter(X_labels[0:20], self.y_pred[0:20], color='blue')
        plt.xticks((X_labels[0:20]))
        plt.yticks(self.y_test[0:20])
        if optimized:
            plt.savefig("files/BayesianRidgeRegressor_" + str(task_id) + "_optimized.png", dpi=100)
            return "BayesianRidgeRegressor_" + str(task_id) + "_optimized.png"
        else:
            plt.savefig("files/BayesianRidgeRegressor_" + str(task_id) + ".png", dpi=100)
            return "BayesianRidgeRegressor_" + str(task_id) + ".png"

    def hyperopt_optimization(self, task_id):
        self.__get_data()

        def define_space():
            space = hp.choice('regressor', [
                {
                    'model': linear_model.BayesianRidge,
                    'param':
                        {
                            'n_iter': hp.choice("n_iter", [100, 200, 300, 400, 500, 600]),
                            'tol': hp.uniform("tol", 0, 2e-3),
                            'compute_score': hp.choice('compute_score', [True, False]),
                            'verbose': hp.choice('verbose', [True, False])
                        }
                }])
            return space

        def optimize(args):
            n_iter = args['param']['n_iter']
            tol = args['param']['tol']
            compute_score = args['param']['compute_score']
            verbose = args['param']['verbose']
            model = linear_model.BayesianRidge(n_iter=n_iter, verbose=verbose,
                                               tol=tol, compute_score=compute_score)
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            # preds=model.predict_proba(self.X_test)
            accuracy = metrics.r2_score(self.y_test, preds)
            return -1.0 * accuracy

        import warnings
        warnings.filterwarnings('ignore')
        optimziation_function = partial(optimize)
        trials = Trials()
        space = define_space()
        result = fmin(
            fn=optimziation_function,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials
        )
        self.best_parameters = space_eval(space, result)
        return self.run_optimized_model(task_id)

    def run_optimized_model(self, task_id):
        n_iter = self.best_parameters['param']['n_iter']
        verbose = self.best_parameters['param']['verbose']
        tol = self.best_parameters['param']['tol']
        compute_score = self.best_parameters['param']['compute_score']
        args = {"n_iter": n_iter, "verbose": verbose, "tol": tol, "compute_score": compute_score}
        return {'training': self.training(args), 'visualize': self.visualize(task_id, optimized=True),
                'best_parameters': self.best_parameters['param']}
