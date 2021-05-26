from src.ml.PreProcessing.preprocessing import PreProcessing
import matplotlib. pyplot as plt
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import numpy as np
import io
import sys

from functools import partial
from hyperopt import hp,fmin,tpe,Trials
from hyperopt import space_eval


#GradientBoostingRegressor

class GBRModel():
	def __init__(self,predicted_column,path,categorical_columns,sheet_name=0,train_test_split=True,supplied_test_set=None,percentage_split=0.2):
		self.predicted_column = predicted_column
		self.categorical_columns=categorical_columns
		self.path = path
		self.sheet_name = sheet_name
		self.train_test_split = train_test_split
		self.supplied_test_set = supplied_test_set
		self.percentage_split = percentage_split
	def __get_data(self,train_test_split=True):
		Preprocess=PreProcessing(self.path,self.sheet_name)
		Preprocess.set_predicted_column(self.predicted_column)
		Preprocess.dropping_operations()
		Preprocess.label_encoding()
		Preprocess.fill_missing_values(self.categorical_columns)
		X_train, X_test, y_train, y_test = Preprocess.train_split_test(supplied_test_set=self.supplied_test_set
																	  , percentage_split=self.percentage_split,
																	  train_test_splitt=self.train_test_split)
		self.X_train=X_train
		self.X_test=X_test
		self.y_train=y_train
		self.y_test=y_test
		return True

		
	def score_estimator(self, y_pred, test_data, predicted_column):
		"""Score an estimator on the test set."""
		old_stdout = sys.stdout
		new_stdout = io.StringIO()
		sys.stdout = new_stdout
		print("MSE: %.3f" %
			  mean_squared_error(test_data, y_pred))
		print("MAE: %.3f" %
			  mean_absolute_error(test_data, y_pred))
		print("Accuracy Of Model", metrics.r2_score( self.y_test,self.y_pred))
		output = new_stdout.getvalue()
		sys.stdout = old_stdout
		return output
	def training(self,args={"n_estimators":100, "max_depth":10, "learning_rate":0.1}):
		  self.__get_data()
		  self.regr = GradientBoostingRegressor(**args)
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
	def visualize(self):
		X_labels = np.arange(len(self.y_test))
		plt.scatter(X_labels[0:15], self.y_test[0:15], color='black')
		plt.scatter(X_labels[0:15], self.y_pred[0:15], color='blue')
		plt.xticks((X_labels[0:15]))
		plt.yticks(self.y_test[0:15])
		plt.figure(figsize=(100, 100))
		plt.savefig("GBR_compared_test_and_prediction.png")
	def hyperopt_optimization(self):
		self.__get_data()
		def define_space():
		  space = hp.choice('regressor',[
						{
						'model': GradientBoostingRegressor,
						'param':
						  {    
							  'n_estimators':hp.choice('n_estimators', range(50,500,50)),
							  # 'loss':hp.choice('loss', ['ls','quantile','huber']),
							  'learning_rate':hp.choice('learning_rate',np.arange(0.1, 1.0, 0.1)),
							  'max_depth':hp.choice('max_depth', range(9, 20, 1)),
						  }
						}])
		  return space
		def optimize(args):
			n_estimators = args['param']['n_estimators']
			learning_rate = args['param']['learning_rate']
			max_depth = args['param']['max_depth']
			model=GradientBoostingRegressor(n_estimators = n_estimators ,#loss =loss,
											learning_rate = learning_rate , 
											max_depth = max_depth)
			model.fit(self.X_train,self.y_train)
			preds=model.predict(self.X_test)
			accuracy=metrics.r2_score(self.y_test,preds)
			return -1.0*accuracy
		import warnings
		warnings.filterwarnings('ignore')
		optimziation_function=partial(optimize)
		trials=Trials()
		space=define_space()
		result=fmin(
			fn=optimziation_function,
			space=space,
			algo=tpe.suggest,
			max_evals=100, 
			trials=trials
		)
		self.best_parameters=space_eval(space,result)
		return self.best_parameters
	def run_optimized_model(self):
		n_estimators = self.best_parameters['param']['n_estimators']
		learning_rate = self.best_parameters['param']['learning_rate']
		max_depth = self.best_parameters['param']['max_depth']
		#loss = self.best_parameters['param']['loss']
		args={"n_estimators" : n_estimators , "learning_rate" : learning_rate ,
			  "max_depth" : max_depth  }
		print(self.training(args))
		self.visualize()