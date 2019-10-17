import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.stats import norm
from abc import ABCMeta, abstractmethod

# Define Spread Option Class

class SpreadOption(object):
    ''' Abstract Base-class for valuation of European Spreads Options (Calls/P.uts).
    S1_t : float : initial forward price/index level
    S2_t : float : initial forward price/index level
    K : float : strike price
    T : float : maturity (in year fractions)
    r : float : constant risk-free short rate
    vol1 : float : volatility factor in diffusion term (std)
    vol2 : float : volatility factor in diffusion term (std)
    rho : float: 
    CallPut : integer : 1 for a Call, and -1 for a Put
    '''
    __metaclass__ = ABCMeta

    def __init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, model):
        try:

            self.S1_t = float(S1_t)
            self.S2_t = float(S2_t)
            self.K = float(K)
            self.T = float(T)
            self.r = float(r)
            self.vol1 = float(vol1)
            self.vol2 = float(vol2)
            self.rho = float(rho)
            self.CallPut = int(CallPut)
            self.model = str(model)

            if T < 0 or r < 0 or S1_t < 0 or S2_t < 0:
                raise ValueError('Negative inputs not allowed.')
            if vol1 < 0 or vol2 < 0:
                raise ValueError('Negative volatilities are not allowed.')
            if rho > 1 or rho < -1:
                raise ValueError('Correlation out of range')
            if CallPut != 1 and CallPut != -1:
                raise ValueError('For a Call: CallPut=1, or -1 for a Put')

        except ValueError:
            print('Error passing spread option inputs')

    def getmodel(self):
        return self.model

    def __str__(self):
        return "This SpreadOption is solved using {0}".format(self.getmodel())

    @abstractmethod
    def price(self):
        pass

# First we model Margrabe Option to replicate Spread Option

class margrabe(SpreadOption):
    def __init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut):
        SpreadOption.__init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, "Margrabe")
        if K != 0:
            raise ValueError('Strike should be null to use Margrabe')

    @property
    def price(self):
        vol = np.sqrt(self.vol1 ** 2 + self.vol2 ** 2 - 2 * self.rho * self.vol1 * self.vol2)
        d1 = (np.log(self.S2_t / self.S1_t) / (vol * np.sqrt(self.T)) + 0.5 * vol * np.sqrt(self.T))
        d2 = d1 - vol * np.sqrt(self.T)
        price = (self.CallPut * (self.S2_t * norm.cdf(self.CallPut * d1, 0, 1)
                                 - self.S1_t * norm.cdf(self.CallPut * d2, 0, 1)))
        return price

SpreadCALL = margrabe(25, 25, 0., .75, 0., .5, .5, .85, 1)
print (SpreadCALL)
print (SpreadCALL.price)

#  We experiment how Margrabe Options behave with changing volatilities

def margrabe_experiment():
    for S1_t in (25., 30., 40.):  # initial stock price values
        for S2_t in (25., 30., 40.):
            print ('-'*90)
            for rho in (-.99, -.75,-.5,-.25, 0, .25, .5, .75, .99):
                SpreadCALL = margrabe(S1_t, S2_t, 0, 1, .0, 0.5, 0.5, rho, 1)
                print ("Initial prices: {0}, Sigmas: {1}, Correlation: {2} --> Option Value: {3:.2f}"
                       .format((S1_t, S2_t), (0.5, 0.5), rho, SpreadCALL.price))

from time import time
t0 = time()
margrabe_experiment()
t1 = time(); d1 = t1 - t0
print ('-'*90)
print (SpreadCALL)
print ("Duration in Seconds {0}".format(d1))

# We Model Kirk´s approximation for Spread Options

class kirk(SpreadOption):
    def __init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut):
        SpreadOption.__init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, "Kirk")

    @property
    def price(self):
        z = self.S1_t / (self.S1_t + self.K * np.exp(-1. * self.r * self.T))
        vol = np.sqrt(self.vol1 ** 2 * z ** 2 + self.vol2 ** 2 - 2 * self.rho * self.vol1 * self.vol2 * z)
        d1 = (np.log(self.S2_t / (self.S1_t + self.K * np.exp(-self.r * self.T)))
              / (vol * np.sqrt(self.T)) + 0.5 * vol * np.sqrt(self.T))
        d2 = d1 - vol * np.sqrt(self.T)
        price = (self.CallPut * (self.S2_t * norm.cdf(self.CallPut * d1, 0, 1)
                                 - (self.S1_t + self.K * np.exp(-self.r * self.T))
                                 * norm.cdf(self.CallPut * d2, 0, 1)))
        return price

SpreadCALL = kirk(40, 40, 0., .75, 0., .5, .5, .85, 1)
print (SpreadCALL)
print (SpreadCALL.price)

# Experiment here also with Volatilities changes

def kirk_experiment():
    for S1_t in (25., 30., 40.):  # initial stock price values
        for S2_t in (25., 30., 40.):
            print ('-'*90)
            for rho in (-.99, -.75,-.5,-.25, 0, .25, .5, .75, .99):
                SpreadCALL = kirk(S1_t, S2_t, 0., 1, .0, 0.5, 0.5, rho, 1)
                print ("Initial prices: {0}, Sigmas: {1}, Correlation: {2} --> Option Value: {3:.2f}"
                       .format((S1_t, S2_t), (0.5, 0.5), rho, SpreadCALL.price))

from time import time
t0 = time()
kirk_experiment()
t1 = time(); d1 = t1 - t0
print ('-'*90)
print (SpreadCALL)
print ("Duration in Seconds %6.3f" % d1)

# Compare the closed formulas for Spread Options against Montecarlo Simulation

class montecarlo(SpreadOption):
    def __init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, simulations):
        SpreadOption.__init__(self, S1_t, S2_t, K, T, r, vol1, vol2, rho, CallPut, "Monte Carlo")
        self.simulations = int(simulations)
        try:
            if self.simulations > 0:
                assert isinstance(self.simulations, int)
        except:
            raise ValueError("Simulation's number has to be positive integer")

    def generate_spreads(self, seed=12345678):
        try:
            if seed is not None:
                assert isinstance(seed, int)
        except:
            print
            'Error passing seed'
        np.random.seed(seed)
        B1 = np.sqrt(self.T) * np.random.randn(self.simulations, 1)
        B2 = np.sqrt(self.T) * np.random.randn(self.simulations, 1)
        S1_T = self.S1_t * np.exp((self.r - 0.5 * self.vol1 ** 2) * self.T + self.vol1 * B1)
        S2_T = self.S2_t * np.exp((self.r - 0.5 * self.vol2 ** 2) * self.T +
                                  self.vol2 * (self.rho * B1 + np.sqrt(1 - self.rho ** 2) * B2))
        if self.CallPut == 1:
            payoff = np.maximum((S2_T - S1_T - self.K), 0)
        else:
            payoff = np.maximum((self.K - S2_T - S1_T), 0)
        return np.exp(-1. * self.r * self.T) * payoff

    @property
    def price(self):
        price = np.sum(self.generate_spreads()) / float(self.simulations)
        return price

print ('-'*50)
SpreadCALL1 = kirk(40, 40, 3., 1., 0., .5, .5, .85, 1)
print (SpreadCALL1)
print (SpreadCALL1.price)
print ('-'*50)
SpreadCALL2 = montecarlo(40, 40, 3., 1., 0., .5, .5, .85, 1, 1000000)
print (SpreadCALL2)
print (SpreadCALL2.price)
print ('-'*50)

def montecarlo_experiment():
    for S1_t in (36., 40., 44.):  # initial stock price values
        for S2_t in (44., 40., 36.):
            print ('-'*90)
            for rho in (.01, .5, .99):
                SpreadCALL = montecarlo(S1_t, S2_t, 0., 1, .0, 0.5, 0.5, rho, 1, 1000000)
                print ("Initial prices: {0}, Sigmas: {1}, Correlation: {2} --> Option Value: {3:.2f}"
                       .format((S1_t, S2_t), (0.5, 0.5), rho, SpreadCALL.price))


# We want to analyze how the performance (time for execution) for closed formulas and Monte Carlo Simulations
# And we Plot the Results and see where they differ.

from time import time
t0 = time()
montecarlo_experiment()
t1 = time(); d1 = t1 - t0
print ('-'*90)
print ("Duration in Seconds %6.3f" % d1)

set_corr = [i / 100. for i in range(99)]
result_kirk = []
result_montecarlo = []
for rho in set_corr:
    SpreadCALL1 = kirk(25, 25, 3., 1., 0., .5, .5, rho, 1)
    result_kirk.append(SpreadCALL1.price)
    SpreadCALL2 = montecarlo(25, 25, 3., 1., 0., .5, .5, rho, 1, 100000)
    result_montecarlo.append(SpreadCALL2.price)

plt.figure(num=None, figsize=(14, 6))
plt.style.use('ggplot')
plt.plot(set_corr, 100 * (np.array(result_montecarlo) - np.array(result_kirk)) / np.array(result_kirk))
plt.title('Relative Differenz zwischen MonteCarlo - Kirk Spreads')
plt.xlabel('Korrelation')
plt.ylabel('% error')
plt.show()

set_strikes = range(-30, 30)
result_kirk = []
result_montecarlo = []
for strike in set_strikes:
    SpreadCALL1 = kirk(25, 25, strike, 1., 0., .5, .5, 0.5, 1)
    result_kirk.append(SpreadCALL1.price)
    SpreadCALL2 = montecarlo(25, 25, strike, 1., 0., .5, .5, 0.5, 1, 100000)
    result_montecarlo.append(SpreadCALL2.price)

plt.figure(num=None, figsize=(14, 6))
plt.style.use('ggplot')
plt.plot(set_strikes, 100 * (np.array(result_montecarlo) - np.array(result_kirk)) / np.array(result_kirk))
plt.title('Relative Differenz zwischen MonteCarlo - Kirk Spreads')
plt.xlabel('Strikes')
plt.ylabel('% error')
plt.show()

vol1 = [i / 100. for i in range(1, 50)]
vol2 = [i / 100. for i in range(50, 1, -1)]
result_kirk = []
result_montecarlo = []
for i in range(len(vol1)):
    SpreadCALL1 = kirk(25, 25, 0, 1., 0., vol1[i], vol2[i], 0.5, 1)
    result_kirk.append(SpreadCALL1.price)
    SpreadCALL2 = montecarlo(25, 25, 0, 1., 0., vol1[i], vol2[i], 0.5, 1, 100000)
    result_montecarlo.append(SpreadCALL2.price)

plt.figure(num=None, figsize=(14, 6))
plt.style.use('ggplot')
dif_vols = [vol1[i] - vol2[i] for i in range(len(vol1))]
plt.plot(dif_vols, 100 * (np.array(result_montecarlo) - np.array(result_kirk)) / np.array(result_kirk))
plt.title('Relative Differenz zwischen MonteCarlo - Kirk Spreads')
plt.xlabel('vol1-vol2')
plt.ylabel('% error')
plt.show()
