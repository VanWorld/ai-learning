import numpy as np


def print_ndarray(arr):
    print(arr)
    print(arr.shape)
    print(arr.ndim)
    print(arr.size)
    print(arr.dtype)
    print(arr.itemsize)
    print(arr.data)
    print()


def my_f(x, y):
    return 10*x + y


if __name__ == "__main__":
    a1 = np.array([(1., 2., 3., ), (3., 4., 6.1, )])
    a2 = np.array([[1., 2., 3., ], [3., 4., 6.1, ]])
    # print(type(a1), type(a2))

    a3 = np.arange(15).reshape(3, 5)
    # print_ndarray(a3)

    # explicitly specified type
    a4 = np.array([[1, 2], [3, 4]], dtype=complex)
    # print_ndarray(a4)

    # create array with placehold
    a5 = np.zeros((3, 4))
    # print_ndarray(a5)
    a6 = np.ones((2, 3, 4), dtype=np.int16)
    # print_ndarray(a6)
    a7 = np.empty((2, 3))
    # print_ndarray(a7)

    # create sequence array
    a8 = np.arange(10, 30, 5)
    # print_ndarray(a8)

    # linspace
    a9 = np.linspace(0, 2, 9)
    # print_ndarray(a9)

    a10 = np.linspace(0, 2*np.pi, 100)  # useful to evaluate function at lots of points
    f = np.sin(a10)
    # print_ndarray(f)

    # arithmetic operation
    a11 = np.array([20, 30, 40, 50])
    a12 = np.arange(4)
    # print_ndarray(a11 + a12)
    # print_ndarray(a11 - a12)
    # print_ndarray(a11**2)
    # print_ndarray(a11 < 33)

    ## product
    a13 = np.array([[1, 1], [0, 1]])
    a14 = np.array([[2, 0], [3, 4]])
    # print(a13 * a14)
    # print(a13 @ a14) # matrix product
    # print(a13.dot(a14)) # matrix product

    # random
    a15 = np.random.random((2, 3))
    # print_ndarray(a15)

    # unary operation
    a16 = np.arange(12).reshape(3, 4)
    # print(a16.sum())
    # print(a16.min())
    # print(a16.max())
    # print(a16.sum(axis=0))  # to understand axis and dimension
    # print(a16.min(axis=1))
    # print(a16.max(axis=0))
    # print(a16.cumsum(axis=0))
    # print(a16[0][0])

    # a17 = np.arange(10)
    # print(a17)
    # a17[:6:2] = 1000
    # print(a17)
    # a17 = a17[::-1]
    # print(a17)
    #
    # a18 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # print(a18)
    # print(a18[::-1])

    # a19 = np.fromfunction(my_f, (5, 4), dtype=int)
    # print(a19)
    # # print(a19[2:5, 0])  # (2:5, 0)是tuple， 每一个元素表示一个axis的index
    #
    # # iterate
    # for row in a19:
    #     print(row)
    # for row in a19[1]:
    #     print(row)
    # for item in a19.flat:
    #     print(item)

    # shape manipulation
    a20 = np.floor(10*np.random.random((3, 4)))
    # print(a20)
    # print(a20.ravel())
    # print(a20.reshape(6, 2))
    # print(a20.T)  # transpose
    #
    # print(a20)
    # print(a20.resize(6, 2))  # resize change the array itself
    # print(a20)

    # view or shallow copy
    # a20_view = a20.view()
    # print(type(a20_view))
    # print(a20_view)
    # print(a20_view is a20)
    # print(a20_view.base is a20)

    # deep copy
    a20_deep_copy = a20.copy()
    print(a20_deep_copy)
    print(a20_deep_copy is a20)
    print(a20_deep_copy.base is a20)