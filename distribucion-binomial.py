import numpy as np
from numpy.random import binomial
from scipy.stats import binom
from math import factorial
import matplotlib.pyplot as plt

#def my_binomial(k, n, p):
    #return factorial(n)/(factorial(k) * factorial(n-k)) * pow(p,k)*pow(1-p,n-k)

#print(my_binomial(2, 3, 0.5))

#dist = binom(3, 0.5)
#print(dist.pmf(2))
#dist.cdf(2)

p = 0.5
n = 3

def plot_hist(num_trials):
    values = [0, 1, 2, 3]
    array = []
    for _ in range(num_trials):
        array.append(binomial(n, p))

    sim = np.unique(array, return_counts=True) [1]/len(array)
    teorica = [binom(3, 0.5).pmf(k) for k in values]
    plt.bar(values, sim, color = 'red')
    plt.bar(values, teorica, alpha = 0.5, color = 'blue')
    plt.title('{} experimentos'.format(num_trials))
    plt.show()

print(plot_hist(20000))