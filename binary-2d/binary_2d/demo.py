import matplotlib.pyplot as plt
import numpy as np

from .functions import fn1

np.random.seed(3)

def main():
    X, y = fn1(n=500, m=500, p=0.6)
    fig, ax = plot(X, y)
    fig.show()

def plot(X, y):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=200, marker='o', c='dimgrey', alpha=0.9, edgecolor='black')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=200, marker='o', c='white', alpha=0.9, edgecolor='black')
    return fig, ax

if __name__ == '__main__':
    main()
