from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from Load_And_Visualize_Time_Data import Load_and_Visualize_Time_Data
import sys
import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.metrics import mean_absolute_error
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import smape_loss
class Exponential_Smoothing_Model:
  def __init__(self,path,time_freq,trend=None,seasonal=None,time_column=0,day_to_month=False,month_to_year=False,test_size=0.2,model="additive"):
    preload=Load_and_Visualize_Time_Data(path,time_column,model)
    self.data,self.columns=preload.return_data()
    preload.visualize_data()
    preload.decompose_time_series()
    if day_to_month and time_freq=='M':
      self.day_to_month()
    elif month_to_year and time_freq=='Y':
      self.day_to_month()
      self.month_to_year()
    else:
      sys.exit("time frequency and converted frequency does not match")
    self.time_freq=time_freq
    self.trend=trend
    self.seasonal=seasonal
    self.test_size=test_size
  def train_test_split(self):
    self.data.index = pd.PeriodIndex(self.data.index, freq=self.time_freq)
    self.y_train, self.y_test = temporal_train_test_split(self.data, test_size=self.test_size)
  def day_to_month(self):
    self.data=self.data.resample('M').sum()
  def month_to_year(self):
    self.data=self.data.resample('Y').sum()
  def best_alpha_value(self,alphas,step):
    best_mae=0
    i=0
    best_alpha=0
    for alpha in alphas:
      ses_model = SimpleExpSmoothing(self.y_train).fit(smoothing_level=alpha)
      y_pred = ses_model.forecast(step)
      mae = mean_absolute_error(self.y_test, y_pred)
      if i ==0:
        best_mae=mae
        i+=1
      elif mae<best_mae:
        best_mae=mae
        best_alpha=alpha
    return best_alpha
  def forecast_and_visualize(self):
    alphas = np.arange(0.01, 1, 0.10)
    alpha=self.best_alpha_value(alphas, len(self.y_test))
    # alpha: 0.11 mae: 82.528

    # Modelin en uygun alpha değeri ile oluşturulması
    forecaster = SimpleExpSmoothing(self.y_train).fit(smoothing_level=alpha)
    fh = np.arange(1, len(self.y_test) + 1)
    # 24 aylık tahmin
    y_pred = forecaster.forecast(len(self.y_test))
    plot_series(self.y_train, self.y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
    print(smape_loss(self.y_test, y_pred))