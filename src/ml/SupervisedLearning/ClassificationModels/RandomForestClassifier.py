from api.algorithms.PreProcessing.preprocessing import PreProcessing
from api.algorithms.Visualization.Visualization_Functions import Visualization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import sys
import io
from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt import space_eval


class RandomForest_Classifier():
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

    def training(self, args={"max_depth": 10, "random_state": 0}):
        self.__get_data()
        self.classification = RandomForestClassifier(**args)
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
            plt.savefig("files/randomForest_" + str(task_id) + "_optimized.png", dpi=100)
            return "randomForest_" + str(task_id) + "_optimized.png"
        else:
            plt.savefig("files/randomForest_" + str(task_id) + ".png", dpi=100)
            return "randomForest_" + str(task_id) + ".png"

    def visualize_classes(self, task_id, optimized=False):
        plt.clf()
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.X_train)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.y_train)
        plt.xlabel('First principle component')
        plt.ylabel('Second principle component')
        if optimized:
            plt.savefig("files/randomForest_classes_" + str(task_id) + "_optimized.png", dpi=100)
            return "randomForest_classes_" + str(task_id) + "_optimized.png"
        else:
            plt.savefig("files/randomForest_classes_" + str(task_id) + ".png", dpi=100)
            return "randomForest_classes_" + str(task_id) + ".png"

    def classification_report(self):
        target_names = self.label_names.astype(str)
        return classification_report(self.y_test, self.y_pred_linear.round(), target_names=target_names,
                                     output_dict=True)

    def hyperopt_optimization(self, task_id):
        self.__get_data()

        def define_space():
            space = hp.choice('classifier', [
                {
                    'model': RandomForestClassifier,
                    'param':
                        {
                            "n_estimators": hp.quniform("n_estimators", 20, 100, 1),
                            "criterion": hp.choice("criterion", ["gini", "entropy"]),
                            "max_depth": hp.quniform("max_depth", 5, 30, 1),
                            "min_samples_split": hp.uniform("min_samples_split", 0.1, 1.0),
                            "min_samples_leaf": hp.uniform("min_samples_leaf", 0.1, 0.5),
                            "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0, 0.5),
                            "max_features_choices": hp.choice("max_features_choices",
                                                              [
                                                                  {
                                                                      "max_features": hp.uniform("max_features_1", 0.1,
                                                                                                 1.0)},
                                                                  {
                                                                      "max_features": hp.choice("max_features_2",
                                                                                                ["auto", "sqrt", "log2",
                                                                                                 None])
                                                                  },
                                                              ]),
                            "bootstrap_choices": hp.choice("bootstrap_choices",
                                                           [
                                                               {
                                                                   "bootstrap": hp.choice("bootstrap_1", [False]),
                                                                   "oob_score": hp.choice("oob_1", [False])
                                                               },
                                                               {
                                                                   "bootstrap": hp.choice("bootstrap_2", [True]),
                                                                   "oob_score": hp.choice("oob_2", [True, False])
                                                               },
                                                           ]),
                            "class_weight": hp.choice("class_weight", ["balanced", "balanced_subsample", None]),
                            "random_state": hp.choice('random_state', [0, 42, None]),
                        }
                }])
            return space

        def optimize(args):
            n_estimators = int(args['param']['n_estimators'])
            criterion = args['param']['criterion']
            max_depth = args['param']['max_depth']
            min_samples_split = args['param']['min_samples_split']
            min_samples_leaf = args['param']['min_samples_leaf']
            min_weight_fraction_leaf = args['param']['min_weight_fraction_leaf']
            max_features = args['param']['max_features_choices']["max_features"]
            bootstrap = args['param']["bootstrap_choices"]['bootstrap']
            oob_score = args['param']["bootstrap_choices"]['oob_score']
            class_weight = args['param']['class_weight']
            random_state = args['param']['random_state']

            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                           min_weight_fraction_leaf=min_weight_fraction_leaf,
                                           max_features=max_features, bootstrap=bootstrap, oob_score=oob_score,
                                           class_weight=class_weight, random_state=random_state, )
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
            max_evals=100,  # bu değer değerlendirilecek
            trials=trials, verbose=False
        )
        self.best_parameters = space_eval(space, result)
        return self.run_optimized_model(task_id)

    def run_optimized_model(self, task_id):
        n_estimators = int(self.best_parameters['param']['n_estimators'])
        criterion = self.best_parameters['param']['criterion']
        max_depth = self.best_parameters['param']['max_depth']
        min_samples_split = self.best_parameters['param']['min_samples_split']
        min_samples_leaf = self.best_parameters['param']['min_samples_leaf']
        min_weight_fraction_leaf = self.best_parameters['param']['min_weight_fraction_leaf']
        max_features = self.best_parameters['param']['max_features_choices']["max_features"]
        bootstrap = self.best_parameters['param']["bootstrap_choices"]['bootstrap']
        oob_score = self.best_parameters['param']["bootstrap_choices"]['oob_score']
        class_weight = self.best_parameters['param']['class_weight']
        random_state = self.best_parameters['param']['random_state']
        args = {"n_estimators": n_estimators, "criterion": criterion, "max_depth": max_depth,
                "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf,
                "min_weight_fraction_leaf": min_weight_fraction_leaf,
                "max_features": max_features, "bootstrap": bootstrap, "oob_score": oob_score,
                "class_weight": class_weight, "random_state": random_state}
        return {'training': self.training(args), 'visualize': self.visualize(task_id, optimized=True),
                'visualize_classes': self.visualize_classes(task_id, optimized=True),
                'classification_report': self.classification_report(),
                'best_parameters': self.best_parameters['param']}
