from sklearn.model_selection import train_test_split
import pandas as pd
import re
import xlrd
from sklearn import preprocessing
import numpy as np

class PreProcessing():

  def __init__(self, path,sheet_name=0):
    self.path = path
    self.sheet_name = sheet_name
    if re.search('.csv$', self.path) is not None:
      self.data=pd.read_csv(self.path)
    elif re.search('.xlsx$', self.path):
      self.data=pd.read_excel(self.path,sheet_name=self.sheet_name,skiprows=range(self.find_skip_rows()))
    elif re.search('.json$', self.path):
      self.data=pd.read_json(self.path)
    elif re.search('.xml$', self.path):
      self.data=pd.read_xml(self.path)
    elif re.search('.html$', self.path):
      self.data=pd.read_html(self.path)
    else:
      raise Exception("Veri Girişinde Hata")

  def set_predicted_column(self,predicted_column):
      self.predicted_column = self.data.columns[predicted_column]
      return self.predicted_column

  def get_column_names(self):
    return self.data.columns

  def get_label_names(self):
    return self.data[self.predicted_column].unique()

  def get_data(self):
    return np.asarray(self.data)

  def sheet_name(self):
    sheet_names=list()
    book = xlrd.open_workbook(self.path)
    for sheet in book.sheets():
      sheet_names.append(sheet.name)
    return sheet_names

  def find_skip_rows(self):
    data=pd.read_excel(self.path)
    for index, row in data.iterrows():
      if row.isnull().any() ==False:
        return index+1

  def dropping_operations(self):
    dropped_columns = []
    dropped_columns_locs = []
    dropped_columns_data = []
    column_counter = 0
    for column in self.data.columns:
      if len(self.data[column].unique()) == len(self.data[column]):
        dropped_columns_data.append(self.data[column])
        dropped_columns_locs.append(column_counter)
        self.data.drop(column, axis=1, inplace=True)
        dropped_columns.append(column)
      elif len(self.data[column].unique()) == 1:
        dropped_columns_data.append(self.data[column])
        dropped_columns_locs.append(column_counter)
        self.data.drop(column, axis=1, inplace=True)
        dropped_columns.append(column)
      column_counter += 1
    return dropped_columns, dropped_columns_data, dropped_columns_locs

  def reverse_dropping_operations(self, dropped_columns, dropped_columns_data, dropped_columns_locs):
    for column_name, column_data, column_loc in zip(dropped_columns, dropped_columns_data, dropped_columns_locs):
      self.data.insert(column_loc, column_name, column_data)
    return self.data


  def label_encoding(self):
    changed_columns=[]
    columns_data=[]
    column_counter=0
    self.le = preprocessing.LabelEncoder()
    dataTypeSeries = self.data.dtypes
    for datatype, column in zip(dataTypeSeries, self.data.columns):
      if datatype == "object":
        changed_columns.append(column)
        columns_data.append(self.data[column])
        self.data[column] = self.le.fit_transform(self.data[column])
      column_counter+=1
    return changed_columns,columns_data

  def reverse_label_encoding(self,changed_columns,columns_data):
    counter=0
    for column in changed_columns:
      self.data = self.data.assign(new=columns_data[counter])
      self.data[column]=self.data["new"]
      self.data.drop(columns=['new'],inplace=True)
      counter+= 1
    return np.asarray(self.data)

  def number_of_records(self):
    self.count_row = self.data.shape[0]
    return self.count_row

  def fill_missing_values(self,categorical_columns):
    #categoricals=self.label_encoding()
    for column in self.data.columns:
      null_count=(self.data[column].isnull().sum()*100)
      if self.data[column].count()==0:
        self.data.drop(column,axis=1,inplace=True)
        continue
      null_vs_count=(self.data[column].isnull().sum()*100)/(self.data[column].count())
      if (null_vs_count)<79 and (null_vs_count)>0 :
        if column in categorical_columns:
          self.data[column] = self.data[column].fillna(self.data[column].value_counts().index[0])
        else:
          self.data.fillna(self.data.mean(),inplace=True)
      elif (null_vs_count)>95:
        self.data.drop(column,axis=1,inplace=True)

  def min_max_scaling(self,X_train,X_test=None):
    self.min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = self.min_max_scaler.fit_transform(X_train)
    if X_test is not None:
      X_test_minmax = self.min_max_scaler.transform(X_test)
    else:
      X_test_minmax=None
    return X_train_minmax,X_test_minmax

  def reverse_min_max_scaling(self):
    return self.min_max_scaler.inverse_transform(self.data)

  def GaussianTranformation(self, X_train, X_test):
    pt = preprocessing.PowerTransformer(method='box-cox', standardize=True)
    return pt.fit_transform(X_train), pt.fit_transform(X_test)


  def Normalization(self, X_train, X_test):
    X_normalized_train = preprocessing.normalize(X_train, norm='l2')
    X_normalized_train = preprocessing.normalize(X_test, norm='l2')
    return X_normalized_train,X_normalized_train

  def train_split_test(self,supplied_test_set=None,percentage_split=0.2,train_test_splitt=True):
    x=self.data.drop(self.predicted_column,axis=1).values
    y=self.data[self.predicted_column].values
    if train_test_splitt:
      X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = percentage_split)
    else:
      y_train=y
      X_train=x
      predicted_column=supplied_test_set.columns[self.predicted_column]
      y_test=supplied_test_set[self.predicted_column]
      X_test=supplied_test_set.drop[self.predicted_column]
    return X_train, X_test, y_train, y_test



