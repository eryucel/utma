import numpy as np
from src.ml.PreProcessing.preprocessing import PreProcessing
from src.ml.Visualization.Visualization_Functions import Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn import metrics, linear_model
import sys
import io
from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt import space_eval


class LogisticRegressionModel():
    def __init__(self, predicted_column, path, categorical_columns, sheet_name=0, train_test_split=True,
                 supplied_test_set=None, percentage_split=0.2):
        self.predicted_column = predicted_column
        self.path = path
        self.categorical_columns = categorical_columns
        self.sheet_name = sheet_name
        self.train_test_split = train_test_split
        self.supplied_test_set = supplied_test_set
        self.percentage_split = percentage_split
        self.sheet_name = sheet_name

    def __get_data(self):
        Preprocess = PreProcessing(self.path, self.sheet_name)
        Preprocess.set_predicted_column(self.predicted_column)
        self.label_names = Preprocess.get_label_names()
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

    def training(self, args={"max_iter": 1000}):
        self.__get_data()
        self.regr = linear_model.LogisticRegression(**args)
        self.regr.fit(self.X_train, self.y_train)
        self.y_pred = self.regr.predict_proba(self.X_test)
        self.y_pred_linear = self.regr.predict(self.X_test)
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        print('Cross-Track error: %.3f'
              % log_loss(self.y_test, self.y_pred))
        print("Accuracy Of Model", metrics.accuracy_score(self.y_test, self.y_pred_linear))
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
        X_labels = np.arange(len(self.y_test))
        # print(X_test1.size,self.y_test.size)
        plt.scatter(X_labels[0:20], self.y_test[0:20], color='black')
        plt.scatter(X_labels[0:20], self.y_pred_linear[0:20], color='blue')
        plt.xticks((X_labels[0:20]))
        plt.yticks(self.y_test[0:20])
        plt.figure(figsize=(50, 50))
        plt.savefig("logreg_compared_test_and_prediction.png")

    def visualize_classes(self):
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.X_train)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.y_train)
        plt.xlabel('First principle component')
        plt.ylabel('Second principle component')
        plt.savefig("classes_logreg.png")

    def classification_report(self):
        target_names = self.label_names.astype(str)
        return classification_report(self.y_test, self.y_pred_linear.round(), target_names=target_names)

    def hyperopt_optimization(self):
        self.__get_data()

        def define_space():
            space = hp.choice('classifier', [
                {
                    'model': linear_model.LogisticRegression,
                    'param':
                        {
                            'hyper_param_groups': hp.choice('hyper_param_groups',
                                                            [
                                                                {
                                                                    'penalty': hp.choice('penalty_block1', ['l2']),
                                                                    'solver': hp.choice('solver_block1',
                                                                                        ['newton-cg', 'sag', 'saga',
                                                                                         'lbfgs']),
                                                                    'multi_class': hp.choice('multi_class',
                                                                                             ['ovr', 'multinomial']),
                                                                },
                                                                {
                                                                    'penalty': hp.choice('penalty_block2', ['l2']),
                                                                    'solver': hp.choice('solver_block2', ['liblinear']),
                                                                    'multi_class': hp.choice('multi_class_block2',
                                                                                             ['ovr']),
                                                                },
                                                                {
                                                                    'penalty': hp.choice('penalty_block3', ['l1']),
                                                                    'solver': hp.choice('solver_block3', ['saga']),
                                                                    'multi_class': hp.choice('multi_class_block3',
                                                                                             ['ovr', 'multinomial']),
                                                                },
                                                            ]),
                            'dual': hp.choice('dual', [False]),
                            'class_weight': hp.choice('class_weight', ['balanced', None]),
                            'random_state': hp.choice('random_state', [0, 42, None]),
                            'max_iter': hp.quniform('max_iter', 100, 500, 1),
                            'verbose': hp.choice('verbose', [0])
                        }
                }])
            return space

        def optimize(args):
            penalty = args['param']['hyper_param_groups']['penalty']
            solver = args['param']['hyper_param_groups']['solver']
            multi_class = args['param']['hyper_param_groups']['multi_class']
            dual = args['param']['dual']
            class_weight = args['param']['class_weight']
            random_state = args['param']['random_state']
            max_iter = args['param']['max_iter']
            verbose = args['param']['verbose']
            model = linear_model.LogisticRegression(penalty=penalty, solver=solver, multi_class=multi_class, dual=dual,
                                                    class_weight=class_weight, random_state=random_state,
                                                    max_iter=max_iter,
                                                    verbose=verbose)
            model.fit(self.X_train, self.y_train)
            # preds=model.predict(self.X_test)
            preds = model.predict_proba(self.X_test)
            # accuracy=metrics.accuracy_score(self.y_test,preds)
            return log_loss(self.y_test, preds)

        import warnings
        warnings.filterwarnings('ignore')
        optimziation_function = partial(optimize)
        trials = Trials()
        space = define_space()
        result = fmin(
            fn=optimziation_function,
            space=space,
            algo=tpe.suggest,
            max_evals=15,  # bu değer değerlendirilecek
            trials=trials
        )
        self.best_parameters = space_eval(space, result)
        return self.best_parameters

    def run_optimized_model(self):
        penalty = self.best_parameters['param']['hyper_param_groups']['penalty']
        solver = self.best_parameters['param']['hyper_param_groups']['solver']
        multi_class = self.best_parameters['param']['hyper_param_groups']['multi_class']
        dual = self.best_parameters['param']['dual']
        class_weight = self.best_parameters['param']['class_weight']
        random_state = self.best_parameters['param']['random_state']
        max_iter = self.best_parameters['param']['max_iter']
        verbose = self.best_parameters['param']['verbose']
        args = {"penalty": penalty, "solver": solver, "multi_class": multi_class, "dual": dual,
                "class_weight": class_weight, "random_state": random_state, "max_iter": max_iter, "verbose": verbose}
        print(self.training(args))
        print(self.classification_report())
        print(self.visualize())