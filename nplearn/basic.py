import numpy as np

if __name__ == "__main__":
    # a1 = np.sort(np.random.rand(40, 1), axis=1)
    # print(a1)
    # a2 = np.sort([0, 100, 100, 234, 23], axis=0)
    # print(a2)

    x = np.array([3, 4, 2, 1])
    r1 = np.argpartition(x, 3)
    print(r1)
    print(type(r1))
    print(x[r1])
