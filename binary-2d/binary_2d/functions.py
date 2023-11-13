import numpy as np

'''
Binary 2D Function 1 ('fn1')

Generates a dataset using two uniform distributions, which possibly overlap.
The first class takes X values in the range [0, p] and Y values in the range [0, 1].
The second class takes X values in the range [q, 1] and Y values in the range [0, 1].

Parameters:
n: The number of patterns to generate in class-0
m: The number of patterns to generate in class-1
p: The upper bound of X values in class-0
q: The lower bound of X values in class-1

Returns:
An (n+m, 2) array representing the dataset
An (n+m,) array representing the class labels
'''
def fn1(n=50, m=50, p=0.2, q=None):
    if q is None:
        q = 1 - p
    c0 = np.random.uniform(low=[0, 0], high=[p, 1], size=(n, 2))
    c1 = np.random.uniform(low=[q, 0], high=[1, 1], size=(m, 2))
    sample = np.append(c0, c1, axis=0)

    y0 = np.full(fill_value=0, shape=(n,))
    y1 = np.full(fill_value=1, shape=(m,))
    y = np.append(y0, y1)
    return sample, y


if __name__ == '__main__':
    print(fn1())
