import matplotlib.pyplot as plt
import numpy as np

from functions import fn1

np.random.seed(3)



def main():
    functions = {'f1': fn1}
    for name, f in functions.items():
        y = f(X)
        fig, ax = plot_pts(X, y, name)
        ax.set_title(name, fontsize=24)
        fig.show()
        fig.savefig(f'out/binary-2d-{name}.png')
        with open(f'out/binary-2d-{name}.csv', 'w') as fh:
            for i in range(len(y)):
                fh.write('%.2f,%.2f,%d\n' % (X[i][0], X[i][1], y[i]))


def generate_pts(n_pts):
    X = np.empty(shape=(n_pts, 2), dtype=float)
    i = 0
    for x1 in np.linspace(0, 1, 11)[:-1]:
        for x2 in np.linspace(0, 1, 11)[:-1]:
            n_grid_pts = n_pts // 100
            grid_pts = np.random.uniform(low=np.array([x1, x2]), high=np.array([x1 + 0.1, x2 + 0.1]),
                                         size=(n_grid_pts, 2))
            for j in range(n_grid_pts):
                X[i] = grid_pts[j]
                i += 1
    return X


def f1(X, noise=0.10):
    # A 4-th degree polynomial which increases from 0 .. 1 (but non-monotonic)
    val = 6.65 * X[:, 0] - 29.5 * X[:, 0] ** 2 + 46.0 * X[:, 0] ** 3 - 22.41 * X[:, 0] ** 4
    adj = np.random.normal(loc=0, scale=noise, size=val.shape)
    y = val + adj < X[:, 1]
    return y.astype(int)


def f2(X, noise=0.05):
    # A circle with diameter 0.75, centered at (0.5, 0.5)
    val = (X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2
    adj = np.random.normal(loc=0, scale=noise, size=val.shape)
    y = val + adj > 0.375 ** 2
    return y.astype(int)


def f3(X, noise=0.10):
    # A disjoint class; the conjunction of two circles:
    # 1 - Centered at (0, 0)    with diameter 0.75
    # 2 - Centered at (1, 0.75) with diameter 1
    val_1 = (X[:, 0]) ** 2 + (X[:, 1]) ** 2
    val_2 = (X[:, 0] - 1) ** 2 + (X[:, 1] - 0.75) ** 2
    adj = np.random.normal(loc=0, scale=noise, size=val_1.shape)
    y1 = (val_1 + adj > 0.375 ** 2)
    y2 = (val_2 + adj > 0.50 ** 2)
    return np.logical_and(y1, y2).astype(int)


def uniform(X, overlap=0.1):
    pass

def plot_pts(X, y, name):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=200, marker='o', c='dimgrey', alpha=0.9, edgecolor='black')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=200, marker='o', c='white', alpha=0.9, edgecolor='black')
    return fig, ax

if __name__ == '__main__':
    main()
