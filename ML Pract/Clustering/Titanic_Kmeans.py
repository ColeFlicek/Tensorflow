import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd

style.use('ggplot')

def Get_categories(dataframe, columns):

    for col in columns:
        dataframe[col] = dataframe[col].astype('category')
        dataframe[col] = dataframe[col].cat.codes
    print(df.head())



df = pd.read_excel('titanic.xls')
df.drop(['name', 'ticket', 'cabin', 'home.dest'], inplace = True)
print(df.head())
df = Get_categories(df, ['sex'])
