import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

style.use('ggplot')
desired_width = 320
pd.set_option('display.width', desired_width)

def Assign_Num_Cats(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = pd.read_excel('titanic.xls')
df.drop(['name', 'ticket', 'cabin', 'home.dest'], 1, inplace = True)
df = Assign_Num_Cats(df)
print(df.head())