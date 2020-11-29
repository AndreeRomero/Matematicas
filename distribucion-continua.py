import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * pow((x-mu)/sigma, 2))
    
x = np.arange(-4, 4, 0.1)
y = gaussian(x, 0, 0.5)
plt.plot(x, y)
#plt.show()

dist = norm(0, 1)
x= np.arange(-4, 4, 0.1)
y= [dist.pdf(value) for value in x]
plt.plot(x, y)
#plt.show()

dist = norm(0, 1)
x= np.arange(-4, 4, 0.1)
y = [dist.cdf(value) for value in x]
plt.plot(x, y)
#plt.show()

df = pd.read_excel('s057.xls')
array = df['Normally Distributed Housefly Wing Lengths'].values[4:]
values, dist = np.unique(array, return_counts=True)
plt.bar(values, dist/len(array))
plt.show()

#estimacion de distribucion

mu = arr.mean()
sigma = arr.std()
x = np.arange(30, 60, 0.1)
dist = norm(mu, sigma)
y = [dist.pdf(value) for value in x]
plt.plot(x, y)
plt.show()