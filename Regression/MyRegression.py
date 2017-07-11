from statistics import mean
import matplotlib.pylab as plt
from matplotlib import style
import numpy as np
from SampleData import Create_Dataset

style.use(['fivethirtyeight', 'ggplot'])

xs, ys = Create_Dataset(100, 50, correlation='neg')

xs = np.array(xs, dtype=np.float64)
ys = np.array(ys, dtype=np.float64)


def Best_Fit_Slope_and_Itercept(xs, ys):
    m = (( (mean(xs) * mean(ys)) - mean(xs * ys) ) / ( mean(xs)**2 - mean(xs**2)))

    b =  mean(ys) - m * mean(xs)
    return m, b

def Squared_Error(ys_orig, ys_line):
    return sum(( ys_line - ys_orig)**2)

def CoD(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_reg = Squared_Error(ys_orig, ys_line)
    squared_error_y_mean = Squared_Error(ys_orig, y_mean_line)
    return (1 - (squared_error_reg / squared_error_y_mean))

m, b = Best_Fit_Slope_and_Itercept( xs, ys)
print(m, b)


regression_line = [(m*x) + b for x in xs]

r_squared = CoD(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.plot(xs, regression_line, color='r')
plt.show()