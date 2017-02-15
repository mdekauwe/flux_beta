#!/usr/bin/env python
"""
Fit a sigmoid to some data

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (15.02.2017)"
__email__ = "mdekauwe@gmail.com"

import numpy as np
from scipy.optimize import curve_fit

def sigmoid(x, x0, k):
    y = 1.0 / (1.0 + np.exp(-k * (x - x0)))

    return y

if __name__ == "__main__":

    import pylab

    xdata = np.array([0.0, 1.0, 3.0, 4.3, 7.0, 8.0, 8.5, 10.0, 12.0])
    ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43, 0.7, 0.89, 0.95, 0.99])

    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    print (popt)

    x = np.linspace(-1, 15, 50)
    y = sigmoid(x, *popt)


    pylab.plot(xdata, ydata, 'o', label='data')
    pylab.plot(x,y, label='fit')
    pylab.ylim(0, 1.05)
    pylab.legend(loc='best', numpoints=1)
    pylab.show()
