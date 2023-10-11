import matplotlib.pyplot as plt
import numpy as np

np.random.seed(3)


def main(n_points=500, noise=0.05):
    pts, cls = generate_pts(n_points, noise)
    plot_pts(pts, cls)
    with open('out/binary-2d.csv', 'w') as f:
        for i in range(1000):
            f.write('%.2f,%.2f,%d\n' % (pts[i][0], pts[i][1], cls[i]))


def plot_pts(pts, cls):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.scatter(pts[cls == 0][:, 0], pts[cls == 0][:, 1], s=200, marker='o', c='dimgrey', alpha=0.9, edgecolor='black')
    ax.scatter(pts[cls == 1][:, 0], pts[cls == 1][:, 1], s=200, marker='o', c='white', alpha=0.9, edgecolor='black')
    fig.savefig('out/binary-2d.png')


def generate_pts(n_pts=1000, noise=0.05):
    X = np.empty(shape=(n_pts, 2), dtype=float)
    y = np.empty(shape=(n_pts,), dtype=bool)

    i = 0
    for x1 in np.linspace(0, 1, 11)[:-1]:
        for x2 in np.linspace(0, 1, 11)[:-1]:
            n_grid_pts = n_pts // 100
            grid_pts = np.random.uniform(low=np.array([x1, x2]), high=np.array([x1+0.1, x2+0.1]), size=(n_grid_pts, 2))
            for j in range(n_grid_pts):
                X[i], y[i] = grid_pts[j], decide_cls(grid_pts[j], noise)
                i += 1

    return X, y


def decide_cls(pt, noise):
    val = 6.65 * pt[0] - 29.5 * pt[0]**2 + 46.0 * pt[0]**3 - 22.41 * pt[0]**4
    adj = np.random.normal(loc=0, scale=noise)
    return 0 if val + adj > pt[1] else 1


if __name__ == '__main__':
    main(n_points=1000,
         noise=0.10)
