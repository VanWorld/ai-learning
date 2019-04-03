# -*- encoding = utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    np.random.seed(19680801)
    N = 100
    r0 = 0.6
    x = 0.9 * np.random.rand(N)
    y = 0.9 * np.random.rand(N)
    print(x, y)
    area = (20 * np.random.rand(N)) ** 2
    c = np.sqrt(area)
    r = np.sqrt(x * x + y * y)
    area1 = np.ma.masked_where(r < r0, area)
    area2 = np.ma.masked_where(r >= r0, area)
    plt.scatter(x, y, s=area1, marker='^', c=c)
    plt.scatter(x, y, s=area2, marker='o', c=c)
    # theta = np.arange(0, np.pi / 2, 0.01)
    # plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))
    plt.show()
