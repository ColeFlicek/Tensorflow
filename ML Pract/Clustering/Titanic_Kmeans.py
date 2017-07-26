import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
style.use('ggplot')

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

df.drop(['body','name', 'home.dest', 'embarked', 'cabin', 'boat'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
df = Assign_Num_Cats(df)

X = np.array(df.drop(['survived'], 1).astype(float))

X = preprocessing.scale(X)
Y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
print(X[1].astype(float))

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediciton = clf.predict(predict_me)

    if prediciton[0] == Y[i]:
        correct += 1

print(correct / len(X))
