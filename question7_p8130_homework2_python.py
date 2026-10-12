import numpy as np
import scipy as sc
from scipy.stats import binom, poisson, norm

# Question 7

rng = np.random.default_rng(207)
B = 100000

# Part A
# PROB Case A
at_least_4 = binom.sf(3, 200, 0.01)

# PROB Case B
at_least_40 = binom.sf(39, 100, 0.30)

# PROB Case C
at_least_35 = poisson.sf(34, mu = 25)

print(at_least_4, at_least_40, at_least_35)

# APPROX Case A
n, p_event = 200, 0.01

lam = n * p_event

approx_a = poisson.sf(3, mu = lam)

print(approx_a)

# APPROX Case B 
sd_b = np.sqrt(100 * 0.30 * 0.70)

approx_b = norm.sf(39.5, 30, sd_b)

print(sd_b, approx_b)

# APPROX Case C
sd_c = np.sqrt(25)

approx_c = norm.sf(34.5, 25, sd_c)

print(sd_c, approx_c)

# ERROR Case A
abs_err_a = abs(at_least_4 - approx_a)

rel_err_a = (abs(approx_a - at_least_4))/at_least_4

print(abs_err_a, rel_err_a)

# ERROR Case B
abs_err_b = abs(at_least_40 - approx_b)

rel_err_b = (abs(approx_b - at_least_40))/at_least_40

print(abs_err_b, rel_err_b)

# ERROR Case C
abs_err_c = abs(at_least_35 - approx_c)

rel_err_c = (abs(approx_c - at_least_35))/at_least_35

print(abs_err_c, rel_err_c)


# Part B
# Case A
z_a = rng.binomial(200, .01, size = B)

print(np.mean(z_a >= 4))

# Case B
z_b = rng.binomial(100, .30, size = B)

print(np.mean(z_b >= 40))

# Case C
z_c = rng.poisson(25, size = B)

print(np.mean(z_c >= 35))

# Part C
n, p_birth = 4000, 0.0001

exact_r = binom.sf(9, n, p_birth)

z_r = rng.binomial(n, p_birth, size = B)

print(exact_r)

print(np.sum(z_r >= 10))

print(B * exact_r)


