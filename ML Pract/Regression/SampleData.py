import random
import numpy as np

def Create_Dataset( datapoints, varience, step = 2, correlation=False):
    val = 1
    ys = []
    for i in range(datapoints):
        y = val + random.randrange(-varience, varience)
        ys.append(y)
        if correlation == 'pos':
            val += step
        elif correlation == 'neg':
            val -= step
    xs = [ i for i in range(len(ys))]

    return np.array( xs, dtype=np.float64), np.array( ys, dtype=np.float64)