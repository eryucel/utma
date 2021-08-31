from api.algorithms.PreProcessing.preprocessing import PreProcessing
from api.algorithms.Visualization.Visualization_Functions import Visualization
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import log_loss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import sys
import io
from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt import space_eval


class AdaBoost_Classifier:
    def __init__(self, predicted_column, path, categorical_columns, sheet_name=0, train_test_split=True,
                 supplied_test_set=None, percentage_split=0.2):
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

    def training(self, args={"n_estimators": 100, "random_state": 0}):
        self.__get_data()
        self.classification = AdaBoostClassifier(**args)
        self.classification.fit(self.X_train, self.y_train)
        self.y_pred = self.classification.predict_proba(self.X_test)
        self.y_pred_linear = self.classification.predict(self.X_test)
        result = {'cross_track': log_loss(self.y_test, self.y_pred),
                  'accuracy': self.classification.score(self.X_test, self.y_test)}
        return result

    def predict(self, *X):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        X = np.asarray(X)
        X = [X]
        print(self.classification.predict(X))
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        return output

    def visualize(self, task_id, optimized=False):
        plt.clf()
        X_labels = np.arange(len(self.y_test))
        # print(X_test1.size,self.y_test.size)
        plt.scatter(X_labels[0:20], self.y_test[0:20], color='black')
        plt.scatter(X_labels[0:20], self.y_pred_linear[0:20], color='blue')
        plt.xticks((X_labels[0:20]))
        plt.yticks(self.y_test[0:20])
        if optimized:
            plt.savefig("files/adaBoost_" + str(task_id) + "_optimized.png", dpi=100)
            return "adaBoost_" + str(task_id) + "_optimized.png"
        else:
            plt.savefig("files/adaBoost_" + str(task_id) + ".png", dpi=100)
            return "adaBoost_" + str(task_id) + ".png"

    def visualize_classes(self, task_id, optimized=False):
        plt.clf()
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.X_train)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.y_train)
        plt.xlabel('First principle component')
        plt.ylabel('Second principle component')
        if optimized:
            plt.savefig("files/adaBoost_classes_" + str(task_id) + "_optimized.png", dpi=100)
            return "adaBoost_classes_" + str(task_id) + "_optimized.png"
        else:
            plt.savefig("files/adaBoost_classes_" + str(task_id) + ".png", dpi=100)
            return "adaBoost_classes_" + str(task_id) + ".png"

    def classification_report(self):
        target_names = self.label_names.astype(str)
        return classification_report(self.y_test, self.y_pred_linear.round(), target_names=target_names,
                                     output_dict=True)

    def hyperopt_optimization(self, task_id):
        self.__get_data()

        def define_space():
            space = hp.choice('classifier', [
                {
                    'model': AdaBoostClassifier,
                    'param':
                        {
                            "n_estimators": hp.quniform("n_estimators", 20, 100, 1),
                            "learning_rate": hp.uniform("learning_rate", 0.1, 1.0),
                            "algorithm": hp.choice("algorithm", ["SAMME", "SAMME.R"]),
                            "random_state": hp.choice('random_state', [0, 42, None]),
                        }
                }])
            return space

        def optimize(args):
            n_estimators = int(args['param']['n_estimators'])
            learning_rate = args['param']['learning_rate']
            algorithm = args['param']['algorithm']
            random_state = args['param']['random_state']

            model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm,
                                       random_state=random_state, )
            model.fit(self.X_train, self.y_train)
            self.y_pred = self.classification.predict_proba(self.X_test)
            return log_loss(self.y_test, self.y_pred)

        import warnings
        warnings.filterwarnings('ignore')
        optimziation_function = partial(optimize)
        trials = Trials()
        space = define_space()
        result = fmin(
            fn=optimziation_function,
            space=space,
            algo=tpe.suggest,
            max_evals=25,  # bu değer değerlendirilecek
            trials=trials,
            verbose=False
        )
        self.best_parameters = space_eval(space, result)

        return self.run_optimized_model(task_id)
        # return self.best_parameters

    def run_optimized_model(self, task_id):
        n_estimators = int(self.best_parameters['param']['n_estimators'])
        learning_rate = self.best_parameters['param']['learning_rate']
        algorithm = self.best_parameters['param']['algorithm']
        random_state = self.best_parameters['param']['random_state']
        args = {"n_estimators": n_estimators, "learning_rate": learning_rate, "algorithm": algorithm,
                "random_state": random_state}
        return {'training': self.training(args), 'visualize': self.visualize(task_id, optimized=True),
                'visualize_classes': self.visualize_classes(task_id, optimized=True),
                'classification_report': self.classification_report(),
                'best_parameters': self.best_parameters['param']}
