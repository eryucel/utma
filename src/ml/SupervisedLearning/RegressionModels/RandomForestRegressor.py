from api.algorithms.PreProcessing.preprocessing import PreProcessing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
from six import StringIO
import io
import numpy as np
from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt import space_eval


class RandomForest_Regressor():
    def __init__(self, predicted_column, path, categorical_columns, sheet_name=0, train_test_split=True,
                 supplied_test_set=None, percentage_split=0.2):
        self.predicted_column = predicted_column
        self.categorical_columns = categorical_columns

        self.path = path
        self.sheet_name = sheet_name
        self.train_test_split = train_test_split
        self.supplied_test_set = supplied_test_set
        self.percentage_split = percentage_split

    def __get_data(self, train_test_split=True):
        Preprocess = PreProcessing(self.path)

        Preprocess.set_predicted_column(self.predicted_column)
        Preprocess.dropping_operations()
        Preprocess.label_encoding()
        Preprocess.fill_missing_values(self.categorical_columns)
        X_train, X_test, y_train, y_test = Preprocess.train_split_test(supplied_test_set=self.supplied_test_set
                                                                       , percentage_split=self.percentage_split,
                                                                       train_test_splitt=self.train_test_split)
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

    def training(self, args={"max_depth": 10, "random_state": 0}):
        self.__get_data()
        self.regr = RandomForestRegressor(**args)
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
            plt.savefig("files/randomForestRegressor_" + str(task_id) + "_optimized.png", dpi=100)
            return "randomForestRegressor_" + str(task_id) + "_optimized.png"
        else:
            plt.savefig("files/randomForestRegressor_" + str(task_id) + ".png", dpi=100)
            return "randomForestRegressor_" + str(task_id) + ".png"

    def hyperopt_optimization(self, task_id):
        self.__get_data()

        def define_space():
            space = hp.choice('regressor', [
                {
                    'model': RandomForestRegressor,
                    'param':
                        {
                            'n_estimators': hp.choice('n_estimators', range(50, 500, 50)),
                            'random_state': hp.choice('random_state', np.arange(0, 10, 1)),
                            'max_depth': hp.choice('max_depth', range(6, 20, 1)),
                        }
                }])
            return space

        def optimize(args):
            n_estimators = args['param']['n_estimators']
            random_state = args['param']['random_state']
            max_depth = args['param']['max_depth']
            # loss = args['param']['loss']
            model = RandomForestRegressor(n_estimators=n_estimators,
                                          random_state=random_state,
                                          max_depth=max_depth)
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
        n_estimators = self.best_parameters['param']['n_estimators']
        random_state = self.best_parameters['param']['random_state']
        max_depth = self.best_parameters['param']['max_depth']
        # loss = self.best_parameters['param']['loss']
        args = {"n_estimators": n_estimators, "random_state": random_state,
                "max_depth": max_depth}
        return {'training': self.training(args), 'visualize': self.visualize(task_id, optimized=True),
                'best_parameters': self.best_parameters['param']}
