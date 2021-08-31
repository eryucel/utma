from api.algorithms.PreProcessing.preprocessing import PreProcessing
from api.algorithms.Visualization.Visualization_Functions import Visualization
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from six import StringIO
import matplotlib.image as mpimg
from sklearn import tree
import pydotplus
import sys
import io
import numpy as np
from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt import space_eval


class DecisionTree():
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
        self.columns = Preprocess.get_column_names()
        self.columns = np.delete(self.columns, np.where(Preprocess.data.columns.values == self.predicted_column)[0])
        Preprocess.set_predicted_column(self.predicted_column)
        self.label_names = Preprocess.get_label_names()
        Preprocess.dropping_operations()
        Preprocess.label_encoding()
        Preprocess.fill_missing_values(self.categorical_columns)
        X_train, X_test, y_train, y_test = Preprocess.train_split_test(supplied_test_set=self.supplied_test_set
                                                                       , percentage_split=self.percentage_split,
                                                                       train_test_splitt=self.train_test_split)
        self.X_train = X_train
        # self.X_test=X_test.astype(int)
        self.X_test = X_test
        self.y_train = y_train
        # self.y_test=y_test.astype(int)
        self.y_test = y_test
        return True

    def training(self, args={"criterion": "entropy", "max_depth": 10}):
        self.__get_data()
        self.regr = DecisionTreeClassifier(**args)
        self.regr.fit(self.X_train, self.y_train)
        self.y_pred = self.regr.predict_proba(self.X_test)
        self.y_pred_tree = self.regr.predict(self.X_test)
        result = {'cross_track': log_loss(self.y_test, self.y_pred),
                  'accuracy': metrics.accuracy_score(self.y_test, self.y_pred_tree)}
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

    def visualize(self, task_id, optimized=False):
        plt.clf()
        dot_data = StringIO()
        filename = "drugtree.png"
        featureNames = self.columns.astype(str)
        targetNames = self.label_names.astype(str)

        out = tree.export_graphviz(self.regr, feature_names=featureNames, out_file=dot_data, class_names=targetNames,
                                   filled=True, special_characters=True, rotate=False)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(filename)
        img = mpimg.imread(filename)
        if optimized:
            plt.savefig("files/decisionTree_" + str(task_id) + "_optimized.png", dpi=100)
            return "decisionTree_" + str(task_id) + "_optimized.png"
        else:
            plt.savefig("files/decisionTree_" + str(task_id) + ".png", dpi=100)
            return "decisionTree_" + str(task_id) + ".png"

    def visualize_classes(self, task_id, optimized=False):
        plt.clf()
        visualize = Visualization()
        x_pca = visualize.Dimension_Reduction_with_PCA(self.X_train)
        plt.scatter(x_pca[:, 0], x_pca[:, 1], c=self.y_train)
        plt.xlabel('First principle component')
        plt.ylabel('Second principle component')
        if optimized:
            plt.savefig("files/decisionTree_classes_" + str(task_id) + "_optimized.png", dpi=100)
            return "decisionTree_classes_" + str(task_id) + "_optimized.png"
        else:
            plt.savefig("files/decisionTree_classes_" + str(task_id) + ".png", dpi=100)
            return "decisionTree_classes_" + str(task_id) + ".png"

    def classification_report(self):
        target_names = self.label_names.astype(str)
        return classification_report(self.y_test, self.y_pred_tree.round(), target_names=target_names,
                                     output_dict=True)

    def hyperopt_optimization(self, task_id):
        self.__get_data()

        def define_space():
            space = hp.choice('classifier', [
                {
                    'model': DecisionTreeClassifier,
                    'param':
                        {
                            "criterion": hp.choice("criterion", ["entropy", "gini"]),
                            "splitter": hp.choice("splitter", ["best", "random"]),
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
                            'random_state': hp.choice('random_state', [0, 42, None]),
                            "min_impurity_decrease": hp.uniform("min_impurity_decrease", 0.0, 1.0),
                            "min_impurity_split": hp.quniform("min_impurity_split", 0, 5, 1),
                            "class_weight": hp.choice("class_weight", ["balanced", None]),
                            # "ccp_alpha":hp.uniform("ccp_alpha",0.0,1.0)
                        }
                }])
            return space

        def optimize(args):
            criterion = args['param']['criterion']
            splitter = args['param']['splitter']
            max_depth = args['param']['max_depth']
            max_depth = int(max_depth)
            min_samples_split = args['param']['min_samples_split']
            min_samples_leaf = args['param']['min_samples_leaf']
            min_weight_fraction_leaf = args['param']['min_weight_fraction_leaf']
            max_features = args['param']["max_features_choices"]['max_features']
            random_state = args['param']['random_state']
            min_impurity_decrease = args['param']['min_impurity_decrease']
            min_impurity_split = args['param']['min_impurity_split']
            class_weight = args['param']['class_weight']
            # ccp_alpha = args['param']['ccp_alpha']

            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           min_weight_fraction_leaf=min_weight_fraction_leaf,
                                           max_features=max_features, random_state=random_state,
                                           min_impurity_decrease=min_impurity_decrease,
                                           min_impurity_split=min_impurity_split,
                                           class_weight=class_weight)  # ,ccp_alpha=ccp_alpha)
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
            max_evals=100,  # bu değer değerlendirilecek
            trials=trials, verbose=False
        )
        self.best_parameters = space_eval(space, result)
        return self.run_optimized_model(task_id)

    def run_optimized_model(self, task_id):
        criterion = self.best_parameters['param']['criterion']
        splitter = self.best_parameters['param']['splitter']
        max_depth = self.best_parameters['param']['max_depth']
        min_samples_split = self.best_parameters['param']['min_samples_split']
        min_samples_leaf = self.best_parameters['param']['min_samples_leaf']
        min_weight_fraction_leaf = self.best_parameters['param']['min_weight_fraction_leaf']
        max_features = self.best_parameters['param']["max_features_choices"]['max_features']
        random_state = self.best_parameters['param']['random_state']
        min_impurity_decrease = self.best_parameters['param']['min_impurity_decrease']
        min_impurity_split = self.best_parameters['param']['min_impurity_split']
        class_weight = self.best_parameters['param']['class_weight']
        # ccp_alpha = self.best_parameters['param']['ccp_alpha']
        args = {"criterion": criterion, "splitter": splitter, "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf, "min_weight_fraction_leaf": min_weight_fraction_leaf,
                "max_features": max_features, "random_state": random_state,
                "min_impurity_decrease": min_impurity_decrease, "min_impurity_split": min_impurity_split,
                "class_weight": class_weight, }
        return {'training': self.training(args), 'visualize': self.visualize(task_id, optimized=True),
                'visualize_classes': self.visualize_classes(task_id, optimized=True),
                'classification_report': self.classification_report(),
                'best_parameters': self.best_parameters['param']}
