import numpy as np
import scipy as sc
from scipy.stats import binom, poisson, norm

# Question 7

rng = np.random.default_rng(207)
B = 100000

# Part A

# Case A
at_least_4 = binom.sf(3, 200, 0.01)

# Case B
at_least_40 = binom.sf(39, 100, 0.30)

# Case C
at_least_35 = poisson.sf(34, mu = 25)

print(at_least_4, at_least_40, at_least_35)

# Case A
n, p_event = 200, 0.01

lam = n * p_event

approx_a = poisson.sf(3, mu = lam)

exact_a = binom.sf()

print(approx_a)

# Part B

# Part C

# Part A


# Question 9

rng = np.random.default_rng(209)

# Part A

# Part B

# Part C

# Part D