import numpy as np
from src.ml.PreProcessing.preprocessing import PreProcessing
from src.ml.Visualization.Visualization_Functions import Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn import metrics, model_selection, linear_model
import sys
import io
from functools import partial
from hyperopt import hp, fmin, tpe
from hpsklearn import HyperoptEstimator, random_forest, svc, knn, ada_boost, gradient_boosting, decision_tree, \
    decision_tree, gaussian_nb, svc, svc_linear, svc_rbf, svc_poly, svc_sigmoid


class AutomatedClassificationModelSelection():
    def __init__(self, predicted_column, path, categorical_columns, sheet_name=None, train_test_split=True,
                 supplied_test_set=None, percentage_split=0.2):
        self.predicted_column = predicted_column
        self.path = path
        self.categorical_columns = categorical_columns
        self.sheet_name = sheet_name
        self.train_test_split = train_test_split
        self.supplied_test_set = supplied_test_set
        self.percentage_split = percentage_split
        self.sheet_name = sheet_name
        self.max_evals = 100;

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
        print(np.any(np.isfinite(self.X_train)))
        self.X_test = X_test
        print(np.any(np.isfinite(self.X_test)))
        self.y_train = y_train
        print(np.any(np.isfinite(self.y_train)))
        self.y_test = y_test
        print(np.any(np.isfinite(self.y_test)))
        return True

    def training(self, ):
        self.__get_data()
        if (len(np.unique(self.y_test)) > 2):
            self.estim = HyperoptEstimator(classifier=hp.choice("utma", [(random_forest('utma.random_forest')),
                                                                         (knn("utma.knn")),
                                                                         (decision_tree("utma.decision_tree")),
                                                                         (gaussian_nb("utma.gaussian_nb")),
                                                                         (hp.choice("svc", [(svc('utma.svc')),
                                                                                            (svc_linear(
                                                                                                "utma.svc_linear")),
                                                                                            (svc_rbf("utma.svc_rbf")),
                                                                                            (svc_poly("utma.svc_poly")),
                                                                                            (svc_sigmoid(
                                                                                                "svc_sigmoid")), ]))]),
                                           algo=tpe.suggest,
                                           max_evals=self.max_evals,
                                           trial_timeout=30)
        else:
            self.estim = HyperoptEstimator(classifier=hp.choice("utma", [(random_forest('utma.random_forest')),
                                                                         (knn("utma.knn")),
                                                                         (ada_boost("utma.ada_boost")),
                                                                         (gradient_boosting("utma.gradient_boosting")),
                                                                         (decision_tree("utma.decision_tree")),
                                                                         (gaussian_nb("utma.gaussian_nb")),
                                                                         (hp.choice("svc", [(svc('utma.svc')),
                                                                                            (svc_linear(
                                                                                                "utma.svc_linear")),
                                                                                            (svc_rbf("utma.svc_rbf")),
                                                                                            (svc_poly("utma.svc_poly")),
                                                                                            (svc_sigmoid(
                                                                                                "svc_sigmoid")), ]))]),
                                           algo=tpe.suggest,
                                           max_evals=self.max_evals,
                                           trial_timeout=30)
        self.estim.fit(self.X_train, self.y_train)

    def results(self):
        acc = self.estim.score(self.X_test, self.y_test)
        print("Accuracy: %.3f" % acc)
        return self.estim.best_model()["learner"]
