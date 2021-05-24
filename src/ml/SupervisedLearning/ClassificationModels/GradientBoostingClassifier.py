from src.ml.PreProcessing.preprocessing import PreProcessing
from src.ml.Visualization.Visualization_Functions import Visualization
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import numpy as np
from sklearn.metrics import classification_report
import sys
import io
from functools import partial
from hyperopt import hp,fmin,tpe,Trials
from hyperopt import space_eval

class GBCModle():
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

    def training(self, train_test_split=True):
        self.__get_data()
        self.classification = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=10,
                                                         random_state=0)
        self.classification.fit(self.X_train, self.y_train)
        self.y_pred = self.classification.predict_proba(self.X_test)
        self.y_pred_linear = self.classification.predict(self.X_test)
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        print('Cross-Track error: %.3f'
              % log_loss(self.y_test, self.y_pred))
        #    print("train Accuracy Of Model",self.regr.score(self.X_train, self.y_train))
        print("Accuracy Of Model", self.classification.score(self.X_test, self.y_test))
        #    print(self.y_test[:10])
        #    print(self.y_pred[:10])
        #    print(self.y_pred_linear[:10])
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        return output

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

    def visualize(self):
        X_labels = np.arange(len(self.y_test))
        # print(X_test1.size,self.y_test.size)
        plt.scatter(X_labels[0:20], self.y_test[0:20], color='black')
        plt.scatter(X_labels[0:20], self.y_pred_linear[0:20], color='blue')
        plt.xticks((X_labels[0:20]))
        plt.yticks(self.y_test[0:20])
        plt.figure(figsize=(50, 50))
        plt.savefig("gnb_compared_test_and_prediction.png")

    def visualize_classes(self):
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.X_train)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.y_train)
        plt.xlabel('First principle component')
        plt.ylabel('Second principle component')
        plt.savefig("classes_gnb.png")

    def classification_report(self):
        target_names = self.label_names.astype(str)
        return classification_report(self.y_test, self.y_pred_linear.round(), target_names=target_names)

    def hyperopt_optimization(self):
        self.__get_data()

        def define_space():
            space = hp.choice('classifier', [
                {
                    'model': GradientBoostingClassifier,
                    'param':
                        {
                            "loss": hp.choice("loss", ["deviance", "exponential"]),
                            "learning_rate": hp.uniform("learning_rate", 0.0000, 0.500000),
                            "n_estimators": hp.quniform("n_estimators", 20, 100, 1),
                            "subsample": hp.uniform("subsample", 0.0, 1.0),
                            "criterion": hp.choice("criterion", ["friedman_mse", "mse", "mae"]),
                            "min_samples_split": hp.uniform("min_samples_split", 0.1, 1.0),
                            "min_samples_leaf": hp.uniform("min_samples_leaf", 0.1, 0.5),
                            "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0, 0.5),
                            "max_depth": hp.quniform("max_depth", 5, 30, 1),
                            "min_impurity_decrease": hp.uniform("min_impurity_decrease", 0.0, 0.5),
                            "random_state": hp.choice('random_state', [0, 42, None]),
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
                            "warm_start": hp.choice("warm_start", [True, False]),
                            "validation_fraction": hp.uniform("validation_fraction", 0.0, 1.0)
                        }
                }])
            return space

        def optimize(args):
            loss = args['param']['loss']
            learning_rate = args['param']['learning_rate']
            n_estimators = int(args['param']['n_estimators'])
            subsample = args['param']['subsample']
            criterion = args['param']['criterion']
            min_samples_split = args['param']['min_samples_split']
            min_samples_leaf = args['param']["min_samples_leaf"]
            min_weight_fraction_leaf = args['param']['min_weight_fraction_leaf']
            max_depth = int(args['param']['max_depth'])
            min_impurity_decrease = args['param']['min_impurity_decrease']
            max_features = args['param']["max_features_choices"]['max_features']
            warm_start = args['param']['warm_start']
            validation_fraction = args['param']['validation_fraction']

            model = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                               subsample=subsample, criterion=criterion,
                                               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                               min_weight_fraction_leaf=min_weight_fraction_leaf,
                                               max_depth=max_depth, min_impurity_decrease=min_impurity_decrease,
                                               max_features=max_features,
                                               warm_start=warm_start, validation_fraction=validation_fraction)

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
            trials=trials
        )
        self.best_parameters = space_eval(space, result)
        return self.best_parameters

    def run_optimized_model(self):
        loss = self.best_parameters['param']['loss']
        learning_rate = self.best_parameters['param']['learning_rate']
        n_estimators = int(self.best_parameters['param']['n_estimators'])
        subsample = self.best_parameters['param']['subsample']
        criterion = self.best_parameters['param']['criterion']
        min_samples_split = self.best_parameters['param']['min_samples_split']
        min_samples_leaf = self.best_parameters['param']["min_samples_leaf"]
        min_weight_fraction_leaf = self.best_parameters['param']['min_weight_fraction_leaf']
        max_depth = int(self.best_parameters['param']['max_depth'])
        min_impurity_decrease = self.best_parameters['param']['min_impurity_decrease']
        max_features = self.best_parameters['param']["max_features_choices"]['max_features']
        warm_start = self.best_parameters['param']['warm_start']
        validation_fraction = self.best_parameters['param']['validation_fraction']
        args = {"loss": loss, "learning_rate": learning_rate, "n_estimators": n_estimators, "subsample": subsample,
                "criterion": criterion,
                "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf,
                "min_weight_fraction_leaf": min_weight_fraction_leaf,
                "max_depth": max_depth, "min_impurity_decrease": min_impurity_decrease, "max_features": max_features,
                "warm_start": warm_start, "validation_fraction": validation_fraction}
        print(self.training(args))
        print(self.classification_report())
        self.visualize()
