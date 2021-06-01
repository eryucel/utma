from rest_framework import viewsets, generics, status
from rest_framework.parsers import FileUploadParser, MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
import json
from datetime import datetime
from .serializers import *
from .models import *
from .algorithms.SupervisedLearning.ClassificationModels.AdaBoostClassifier import *
from .algorithms.SupervisedLearning.ClassificationModels.DecisionTree import *
from .algorithms.SupervisedLearning.ClassificationModels.GaussianNaiveBayes import *
from .algorithms.SupervisedLearning.ClassificationModels.GradientBoostingClassifier import *
from .algorithms.SupervisedLearning.ClassificationModels.KNN import *
from .algorithms.SupervisedLearning.ClassificationModels.Logistic_Regression import *
from .algorithms.SupervisedLearning.ClassificationModels.RandomForestClassifier import *
from .algorithms.SupervisedLearning.ClassificationModels.SVMClassification import *
from .algorithms.SupervisedLearning.RegressionModels.AdaBoostRegressor import *
from .algorithms.SupervisedLearning.RegressionModels.BayesianRigdeRegression import *
from .algorithms.SupervisedLearning.RegressionModels.GradientBoostingRegressor import *
from .algorithms.SupervisedLearning.RegressionModels.LassoRegression import *
from .algorithms.SupervisedLearning.RegressionModels.LinearRegression import *
from .algorithms.SupervisedLearning.RegressionModels.RandomForestRegressor import *
from .algorithms.SupervisedLearning.RegressionModels.SGDRegression import *
from .algorithms.UnsupervisedLearning.DBSCAN import *
from .algorithms.UnsupervisedLearning.Hierarchical_Clustering import *
from .algorithms.UnsupervisedLearning.K_Means import *
from .algorithms.Automated_Model_Selection.Automated_Classification_Model_Selection import *
from .algorithms.Automated_Model_Selection.Automated_Regression_Model_Selection import *


# Create your views here.

class RunAlgorithm(APIView):

    def post(self, request, pk):
        task = Task.objects.get(pk=pk)
        serializer = TaskSerializer(task)
        algorithm = serializer.data['algorithm']
        parameters = json.loads(serializer.data['parameters'])
        task_id = serializer.data['id']
        if hasattr(parameters, 'train_test_split'):
            train_test_split = parameters['train_test_split']
        else:
            train_test_split = True
        if hasattr(parameters, 'supplied_test_set'):
            supplied_test_set = parameters['supplied_test_set']
        else:
            supplied_test_set = None
        if hasattr(parameters, 'percentage_split'):
            percentage_split = parameters['percentage_split']
        else:
            percentage_split = 0.2
        result = {}

        if algorithm == 1:
            obj = AdaBoost_Classifier(predicted_column=parameters['predicted_column'],
                                      path='files/' + parameters['dataset'],
                                      categorical_columns=parameters['categorical_columns'].split(','),
                                      train_test_split=train_test_split,
                                      supplied_test_set=supplied_test_set,
                                      percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id),
                                  'visualize_classes': obj.visualize_classes(task_id),
                                  'classification_report': obj.classification_report()},
                      'optimized': obj.hyperopt_optimization(task_id)}
        elif algorithm == 2:
            obj = DecisionTree(predicted_column=parameters['predicted_column'],
                               path='files/' + parameters['dataset'],
                               categorical_columns=parameters['categorical_columns'].split(','),
                               train_test_split=train_test_split,
                               supplied_test_set=supplied_test_set,
                               percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id),
                                  'visualize_classes': obj.visualize_classes(task_id),
                                  'classification_report': obj.classification_report()},
                      'optimized': obj.hyperopt_optimization(task_id)}
        elif algorithm == 3:
            obj = GaussianNaiveBayes(predicted_column=parameters['predicted_column'],
                                     path='files/' + parameters['dataset'],
                                     categorical_columns=parameters['categorical_columns'].split(','),
                                     train_test_split=train_test_split,
                                     supplied_test_set=supplied_test_set,
                                     percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id),
                                  'visualize_classes': obj.visualize_classes(task_id),
                                  'classification_report': obj.classification_report()}}

        elif algorithm == 4:
            obj = GBCModle(predicted_column=parameters['predicted_column'],
                           path='files/' + parameters['dataset'],
                           categorical_columns=parameters['categorical_columns'].split(','),
                           train_test_split=train_test_split,
                           supplied_test_set=supplied_test_set,
                           percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id),
                                  'visualize_classes': obj.visualize_classes(task_id),
                                  'classification_report': obj.classification_report()},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 5:
            obj = KNearestNeighbors(predicted_column=parameters['predicted_column'],
                                    path='files/' + parameters['dataset'],
                                    categorical_columns=parameters['categorical_columns'].split(','),
                                    train_test_split=train_test_split,
                                    supplied_test_set=supplied_test_set,
                                    percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id),
                                  'visualize_classes': obj.visualize_classes(task_id),
                                  'classification_report': obj.classification_report()},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 6:
            obj = LogisticRegressionModel(predicted_column=parameters['predicted_column'],
                                          path='files/' + parameters['dataset'],
                                          categorical_columns=parameters['categorical_columns'].split(','),
                                          train_test_split=train_test_split,
                                          supplied_test_set=supplied_test_set,
                                          percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id),
                                  'visualize_classes': obj.visualize_classes(task_id),
                                  'classification_report': obj.classification_report()},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 7:
            obj = RandomForest_Classifier(predicted_column=parameters['predicted_column'],
                                          path='files/' + parameters['dataset'],
                                          categorical_columns=parameters['categorical_columns'].split(','),
                                          train_test_split=train_test_split,
                                          supplied_test_set=supplied_test_set,
                                          percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id),
                                  'visualize_classes': obj.visualize_classes(task_id),
                                  'classification_report': obj.classification_report()},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 8:
            obj = SvmClassification(predicted_column=parameters['predicted_column'],
                                    path='files/' + parameters['dataset'],
                                    categorical_columns=parameters['categorical_columns'].split(','),
                                    train_test_split=train_test_split,
                                    supplied_test_set=supplied_test_set,
                                    percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id),
                                  'visualize_classes': obj.visualize_classes(task_id),
                                  'classification_report': obj.classification_report()}}

        elif algorithm == 9:
            obj = AdaBoost_Regressor(predicted_column=parameters['predicted_column'],
                                     path='files/' + parameters['dataset'],
                                     categorical_columns=parameters['categorical_columns'].split(','),
                                     train_test_split=train_test_split,
                                     supplied_test_set=supplied_test_set,
                                     percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id)},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 10:
            obj = BayesianRigdeRegressionModel(predicted_column=parameters['predicted_column'],
                                               path='files/' + parameters['dataset'],
                                               categorical_columns=parameters['categorical_columns'].split(','),
                                               train_test_split=train_test_split,
                                               supplied_test_set=supplied_test_set,
                                               percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id)},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 11:
            obj = GBRModel(predicted_column=parameters['predicted_column'],
                           path='files/' + parameters['dataset'],
                           categorical_columns=parameters['categorical_columns'].split(','),
                           train_test_split=train_test_split,
                           supplied_test_set=supplied_test_set,
                           percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id)},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 12:
            obj = LassoRegressionModel(predicted_column=parameters['predicted_column'],
                                       path='files/' + parameters['dataset'],
                                       categorical_columns=parameters['categorical_columns'].split(','),
                                       train_test_split=train_test_split,
                                       supplied_test_set=supplied_test_set,
                                       percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id)},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 13:
            obj = LinearRegressionModel(predicted_column=parameters['predicted_column'],
                                        path='files/' + parameters['dataset'],
                                        categorical_columns=parameters['categorical_columns'].split(','),
                                        train_test_split=train_test_split,
                                        supplied_test_set=supplied_test_set,
                                        percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id)},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 14:
            obj = RandomForest_Regressor(predicted_column=parameters['predicted_column'],
                                         path='files/' + parameters['dataset'],
                                         categorical_columns=parameters['categorical_columns'].split(','),
                                         train_test_split=train_test_split,
                                         supplied_test_set=supplied_test_set,
                                         percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id)},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 15:
            obj = SGDRegressionModel(predicted_column=parameters['predicted_column'],
                                     path='files/' + parameters['dataset'],
                                     categorical_columns=parameters['categorical_columns'].split(','),
                                     train_test_split=train_test_split,
                                     supplied_test_set=supplied_test_set,
                                     percentage_split=percentage_split)
            result = {'default': {'training': obj.training(), 'visualize': obj.visualize(task_id)},
                      'optimized': obj.hyperopt_optimization(task_id)}

        elif algorithm == 16:
            obj = DBSCAN_Model(path='files/' + parameters['dataset'],
                               categorical_columns=parameters['categorical_columns'].split(','))
            result = {
                'default': {'visualize': obj.visualize(task_id), 'clustered_data': obj.return_clustered_data(task_id)}}

        elif algorithm == 17:
            obj = Hierarchical_Clustering_Model(path='files/' + parameters['dataset'],
                                                categorical_columns=parameters['categorical_columns'].split(','),
                                                n_cluster=parameters['n_cluster'])
            result = {
                'default': {'visualize': obj.visualize(task_id), 'clustered_data': obj.return_clustered_data(task_id)}}

        elif algorithm == 18:
            obj = K_MeansModel(path='files/' + parameters['dataset'],
                               categorical_columns=parameters['categorical_columns'].split(','))
            result = {'default': {'visualize': obj.visualize(task_id),
                                  'best_k': obj.best_k_value(), 'visualize_finalize': obj.visualize1(task_id),
                                  'clustered_data': obj.return_clustered_data(task_id)}}

        elif algorithm == 19:
            obj = AutomatedClassificationModelSelection(predicted_column=parameters['predicted_column'],
                                                        path='files/' + parameters['dataset'],
                                                        categorical_columns=parameters['categorical_columns'].split(
                                                            ','),
                                                        train_test_split=train_test_split,
                                                        supplied_test_set=supplied_test_set,
                                                        percentage_split=percentage_split)
            result = {'default': obj.results()}

        elif algorithm == 20:
            obj = AutomatedRegressionModelSelection(predicted_column=parameters['predicted_column'],
                                                    path='files/' + parameters['dataset'],
                                                    categorical_columns=parameters['categorical_columns'].split(','),
                                                    train_test_split=train_test_split,
                                                    supplied_test_set=supplied_test_set,
                                                    percentage_split=percentage_split)
            result = {'default': obj.results()}

        result_serializer = ResultSerializer(
            data={'task': serializer.data['id'], 'algorithmName': serializer.data['algorithmName'],
                  'datasetName': serializer.data['datasetName'], 'data': json.dumps(result)})
        result_serializer.is_valid(raise_exception=True)
        result_serializer.save()
        tmp = dict(serializer.data)
        tmp['status'] = 1
        tmp['completed_date'] = datetime.now()
        serializer = TaskSerializer(task, data=tmp)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(status=status.HTTP_200_OK)


class FileUploadView(APIView):
    parser_class = FileUploadParser

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_serializer = FileSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DatasetCreateApi(generics.CreateAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    # def pre_save(self, obj):
    #     obj.data = self.request.FILES.get('file')


class DatasetListApi(generics.ListAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer


class DatasetUpdateApi(generics.RetrieveUpdateAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer


class DatasetDeleteApi(generics.DestroyAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer


class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer


class TaskCreateApi(generics.CreateAPIView):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer


class TaskListApi(generics.ListAPIView):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer


class ResultRetrieveApi(generics.ListAPIView):
    serializer_class = ResultSerializer

    def get_queryset(self):
        task = self.kwargs['task']
        return Result.objects.filter(task=task)
