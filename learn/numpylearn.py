# -*- encoding = utf-8 -*-

import numpy as np

if __name__ == '__main__':
    ar = np.arange(0., 5., 0.2)
    print(ar)
    print(np.random.randint(3, size=2))

    print(np.zeros((3, 4)))
    print(np.ones((3, 4), dtype=np.int16))
    print(np.empty((2, 3)))

    print(np.random.permutation(100))
