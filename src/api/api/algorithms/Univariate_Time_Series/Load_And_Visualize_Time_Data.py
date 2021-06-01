import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


class Load_and_Visualize_Time_Data:
    def __init__(self, path, time_column, model):
        self.data = pd.read_csv(path, index_col=time_column, parse_dates=True)
        self.model = model

    def return_data(self):
        self.columns = []
        self.columns.append(self.data.columns[0])
        self.columns.append(self.data.index.name)
        self.data = self.data[self.data.columns[0]].squeeze()
        return self.data, self.columns

    def visualize_data(self):
        self.data.plot()
        plt.xlabel(self.columns[0])
        plt.ylabel(self.columns[1])
        plt.show()

    def isstationary(self):
        result = adfuller(self.data)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        if result[1] > 0.5:
            print("This data is not stationary.We recommend you to use stationary data!")
            return False
        print("This data is staionary.")
        return True

    def decompose_time_series(self):
        # additive or multiplicative
        # s_dec_training = sd(data_train.newspx, model = 'additive', freq = 'b')
        result = seasonal_decompose(self.data, model=self.model)
        fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        axes[0].set_title("Decomposition for " + self.model + " model")
        axes[0].plot(self.data, 'k', label='Original ' + self.model)
        axes[0].legend(loc='upper left')

        axes[1].plot(result.trend, label='Trend')
        axes[1].legend(loc='upper left')

        axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
        axes[2].legend(loc='upper left')

        axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
        axes[3].legend(loc='upper left')
        plt.show()
