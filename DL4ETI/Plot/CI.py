import numpy as np
import scipy as sp
import scipy.stats
import numpy


def confidenceinterval(array):
    confidence = 0.95
    a = 1.0 * np.array(array)
    m = np.mean(a)
    fc = scipy.stats.sem(a)
    h = fc * sp.stats.t._ppf((1 + confidence) / 2., n - 1) / ((n - 1) ** 0.5)
    return m - h, m + h


n = 10
a = [71,59,58]
array = numpy.asarray(a)
print(confidenceinterval(array))
