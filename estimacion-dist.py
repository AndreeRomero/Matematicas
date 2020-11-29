import numpy as np
from scipy.stats import norm
from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack
from sklearn.neighbors import KernelDensity

sample = normal(size= 10000) #generador
pyplot.hist(sample, bins = 30)
pyplot.show()

#Estimación paramétrica

sample = normal(loc = 50, scale = 5, size = 10000) # loc == mu; scale == sigma
mu = sample.mean()
sigma = sample.std()
dist = norm(mu, sigma)
values = [value for value in range(30, 70)]
probabilidades = [dist.pdf(value) for value in values]
pyplot.hist(sample, bins = 30, density = True)
pyplot.plot(values, probabilidades)
pyplot.show()

#Estimación no paramétrica: construimos una dist bimodal

sample_1 = normal(loc = 20, scale = 5, size = 300)
sample_2 = normal(loc = 40, scale = 5, size = 700)
sample_fusion = hstack((sample_1, sample_2))

model = KernelDensity(bandwidth= 2, kernel= 'gaussian')
sample_fusion = sample_fusion.reshape((len(sample_fusion), 1))
model.fit(sample_fusion)

values_1 = np.array([value for value in range(1, 60)])
values_1 = values_1.reshape((len(values_1), 1))
probabilities = model.score_samples(values) #probabilidad logaritmica
probabilities = np.exp(probabilities) #inversion de probabilidad

pyplot.hist(sample_fusion, bins = 50, density = True)
pyplot.plot(values, probabilities)
pyplot.show()