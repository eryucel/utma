from statsmodels.tsa.holtwinters import ExponentialSmoothing
from Load_And_Visualize_Time_Data import Load_and_Visualize_Time_Data
import sys
import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.metrics import mean_absolute_error
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import smape_loss
import itertools


class Triple_Exponential_Smoothing:
    def __init__(self, path, time_freq, trend="add", seasonal="add", time_column=0, sp=12, day_to_month=False,
                 month_to_year=False, test_size=0.2, model="additive"):
        preload = Load_and_Visualize_Time_Data(path, time_column, model)
        self.data, self.columns = preload.return_data()
        preload.visualize_data()
        preload.decompose_time_series()
        if day_to_month and time_freq == 'M':
            self.day_to_month()
        elif month_to_year and time_freq == 'Y':
            self.day_to_month()
            self.month_to_year()
        else:
            sys.exit("time frequency and converted frequency does not match")
        self.time_freq = time_freq
        self.trend = trend
        self.seasonal = seasonal
        self.test_size = test_size
        self.sp = sp

    def train_test_split(self):
        self.data.index = pd.PeriodIndex(self.data.index, freq=self.time_freq)
        self.y_train, self.y_test = temporal_train_test_split(self.data, test_size=self.test_size)

    def day_to_month(self):
        self.data = self.data.resample('M').sum()

    def month_to_year(self):
        self.data = self.data.resample('Y').sum()

    def best_alpha__beta_value(self, abg, step):
        results = []
        for comb in abg:
            tes_model = ExponentialSmoothing(self.y_train, trend=self.trend, seasonal=self.seasonal,
                                             seasonal_periods=self.sp).fit(smoothing_level=comb[0],
                                                                           smoothing_slope=comb[1],
                                                                           smoothing_seasonal=comb[2])
            y_pred = tes_model.forecast(step)
            mae = mean_absolute_error(self.y_test, y_pred)
            results.append([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])
        results = pd.DataFrame(results, columns=["alpha", "beta", "gamma", "mae"]).sort_values("mae")
        best_alpha, best_beta, best_gamma, best_mae = results.iloc[0]
        return best_alpha, best_beta, best_gamma

    def forecast_and_visualize(self):
        alphas = betas = gammas = np.arange(0.10, 1, 0.20)
        abg = list(itertools.product(alphas, betas, gammas))
        alpha, beta, gamma = self.best_alpha__beta_value(abg, len(self.y_test))
        # alpha: 0.11 mae: 82.528

        # Modelin en uygun alpha değeri ile oluşturulması
        forecaster = ExponentialSmoothing(self.y_train, trend=self.trend, seasonal=self.seasonal,
                                          seasonal_periods=self.sp).fit(smoothing_level=alpha, smoothing_slope=beta,
                                                                        smoothing_seasonal=gamma)
        fh = np.arange(1, len(self.y_test) + 1)
        # 24 aylık tahmin
        y_pred = forecaster.forecast(len(self.y_test))
        plot_series(self.y_train, self.y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
        print("Loss is:", smape_loss(self.y_test, y_pred))
