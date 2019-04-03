# -*- encoding = utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # fig = plt.figure()
    # fig.suptitle('No Axis on this figure')
    # fig, ax_lst = plt.subplots(2, 2)
    # # plt.show()

    x = np.linspace(0, 2, 100)
    print(x)

    plt.plot(x, x, label='linear')
    plt.plot(x, x**2, label="quadratic")
    plt.plot(x, x**3, label='cubic')

    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title('Sample Plot')
    plt.legend()
    plt.show()

