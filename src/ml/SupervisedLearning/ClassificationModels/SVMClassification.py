from api.algorithms.PreProcessing.preprocessing import PreProcessing
from api.algorithms.Visualization.Visualization_Functions import Visualization
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import sys
import io
from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt import space_eval


class SvmClassification():
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

    def training(self, args={"random_state": 1}):
        self.__get_data()
        self.regr = SVC(**args)
        self.regr.fit(self.X_train, self.y_train)
        self.y_pred = self.regr.decision_function(self.X_test)
        self.y_pred_linear = self.regr.predict(self.X_test)
        result = {'cross_track': metrics.hinge_loss(self.y_test, self.y_pred),
                  'accuracy': metrics.accuracy_score(self.y_test, self.y_pred_linear)}
        return result

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

    def visualize_classes(self, task_id, optimized=False):
        plt.clf()
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.X_train)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.y_train)
        plt.xlabel('First principle component')
        plt.ylabel('Second principle component')
        if optimized:
            plt.savefig("files/SVM_" + str(task_id) + "_optimized.png", dpi=100)
            return "SVM_" + str(task_id) + "_optimized.png"
        else:
            plt.savefig("files/SVM_" + str(task_id) + ".png", dpi=100)
            return "SVM_" + str(task_id) + ".png"

    def visualize(self, task_id, optimized=False):
        plt.clf()
        X_labels = np.arange(len(self.y_test))
        # print(X_test1.size,self.y_test.size)
        plt.scatter(X_labels[0:20], self.y_test[0:20], color='black')
        plt.scatter(X_labels[0:20], self.y_pred_linear[0:20], color='blue')
        plt.xticks((X_labels[0:20]))
        plt.yticks(self.y_test[0:20])
        if optimized:
            plt.savefig("files/SVM_classes_" + str(task_id) + "_optimized.png", dpi=100)
            return "SVM_classes_" + str(task_id) + "_optimized.png"
        else:
            plt.savefig("files/SVM_classes_" + str(task_id) + ".png", dpi=100)
            return "SVM_classes_" + str(task_id) + ".png"

    def classification_report(self):
        target_names = self.label_names.astype(str)
        return classification_report(self.y_test, self.y_pred_linear.round(), target_names=target_names,
                                     output_dict=True)

    # def hyperopt_optimization(self):
    #     self.__get_data()
    #
    #     def define_space():
    #         space = hp.choice('classifier', [
    #             {
    #                 'model': SVC,
    #                 'param':
    #                     {
    #                         "C": hp.uniform("C", 0.1, 1.0),
    #                         "kernel": hp.choice("kernel", ["linear", "poly", "rbf", "sigmoid"]),
    #                         "degree": hp.quniform("degree", 1, 3, 1),
    #                         "gamma": hp.choice("gamma", ["auto", "scale"]),
    #                         "coef0": hp.uniform("coef0", 0.0, 1.0),
    #                         "shrinking": hp.choice("shrinking", [True, False]),
    #                         "probability": hp.choice("probability", [True, False]),
    #                         "class_weight": hp.choice("class_weight", ["balanced", None]),
    #                         "decision_function_shape": hp.choice("decision_function_shape", ["ovo", "ovr"]),
    #                         "break_ties": hp.choice("break_ties", [True, False]),
    #                         "random_state": hp.choice('random_state', [0, 42, None]),
    #                     }
    #             }])
    #         return space
    #
    #     def optimize(args):
    #         C = args['param']['C']
    #         kernel = args['param']['kernel']
    #         degree = args['param']['degree']
    #         gamma = args['param']['gamma']
    #         coef0 = args['param']['coef0']
    #         shrinking = args['param']['shrinking']
    #         probability = args['param']['probability']
    #         class_weight = args['param']['class_weight']
    #         decision_function_shape = args['param']['decision_function_shape']
    #         break_ties = args['param']['break_ties']
    #         random_state = args['param']['random_state']
    #
    #         model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
    #                     probability=probability,
    #                     class_weight=class_weight, decision_function_shape=decision_function_shape,
    #                     break_ties=break_ties, random_state=random_state, )
    #         model.fit(self.X_train, self.y_train)
    #         self.y_pred = self.regr.decision_function(self.X_test)
    #         self.y_pred_linear = self.regr.predict(self.X_test)
    #         return metrics.hinge_loss(self.y_test, self.y_pred)
    #
    #     import warnings
    #     warnings.filterwarnings('ignore')
    #     optimziation_function = partial(optimize)
    #     trials = Trials()
    #     space = define_space()
    #     result = fmin(
    #         fn=optimziation_function,
    #         space=space,
    #         algo=tpe.suggest,
    #         max_evals=10,  # bu değer değerlendirilecek
    #         trials=trials
    #     )
    #     self.best_parameters = space_eval(space, result)
    #     return self.best_parameters
    #
    # def run_optimized_model(self):
    #     C = self.best_parameters['param']['C']
    #     kernel = self.best_parameters['param']['kernel']
    #     degree = self.best_parameters['param']['degree']
    #     gamma = self.best_parameters['param']['gamma']
    #     coef0 = self.best_parameters['param']['coef0']
    #     shrinking = self.best_parameters['param']['shrinking']
    #     probability = self.best_parameters['param']['probability']
    #     class_weight = self.best_parameters['param']['class_weight']
    #     decision_function_shape = self.best_parameters['param']['decision_function_shape']
    #     break_ties = self.best_parameters['param']['break_ties']
    #     random_state = self.best_parameters['param']['random_state']
    #     args = {"C": C, "kernel": kernel, "degree": degree, "gamma": gamma, "coef0": coef0, "shrinking": shrinking,
    #             "probability": probability,
    #             "class_weight": class_weight, "decision_function_shape": decision_function_shape,
    #             "break_ties": break_ties, "random_state": random_state}
    #     print(self.training(args))
    #     print(self.classification_report())
    #     print(self.visualize())
