import numpy as np
from scipy.stats import chi2, t, norm
import matplotlib.pyplot as plt

# Question 9

rng = np.random.default_rng(209)
B = 100000

# Part A

# v = 5
zmat_5 = rng.normal(size = (B, 5))

q_5 = np.sum(zmat_5**2, axis = 1)

print(q_5.mean(), q_5.var(ddof=1))

z0_5 = rng.normal(size = B)

t_5 = z0_5 / np.sqrt(q_5 / 5)

print(t_5.mean(), t_5.var(ddof=1))

# v = 30
zmat_30 = rng.normal(size = (B, 30))

q_30 = np.sum(zmat_30**2, axis = 1)

print(q_30.mean(), q_30.var(ddof=1))

z0_30 = rng.normal(size = B)

t_30 = z0_30 / np.sqrt(q_30 / 30)

print(t_30.mean(), t_30.var(ddof=1))

# Part B

# v = 5
cutoff_5 = chi2.ppf(.95, df = 5)

print(cutoff_5, np.mean(q_5 > cutoff_5))

print(np.mean(np.abs(t_5) > 1.96), 2 * t.sf(1.96, df = 5))

# v = 30
cutoff_30 = chi2.ppf(.95, df = 30)

print(cutoff_30, np.mean(q_30 > cutoff_30))

print(np.mean(np.abs(t_30) > 1.96), 2 * t.sf(1.96, df = 30))

# Part C
plt.hist(q_5, bins=50, density=True)

x = np.linspace(0, 25, 500)

plt.plot(x, chi2.pdf(x, df=5))

plt.show()

plt.hist(t_5, bins=100, range=(-8, 8), density=True)

x = np.linspace(-8, 8, 500)

plt.plot(x, t.pdf(x, df=5))

plt.plot(x, norm.pdf(x))

plt.show()

# Part D

print(chi2.sf(5.6, 3))
