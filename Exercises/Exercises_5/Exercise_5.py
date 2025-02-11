import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import random
from scipy.optimize import brentq
from scipy import optimize

# Problem 1
a = [9.8, 21.2, 34.5, 39.9, 48.5]
sigma_a = [1.0, 1.9, 3.1, 3.9, 5.1]
F = [1., 2., 3., 4., 5.]
sigma_F = [0., 0., 0., 0., 0.]

def pdf(F, m):
    return F/m

# python fit

estimate, error = optimize.curve_fit(pdf, F, a, sigma = sigma_a, absolute_sigma=True)
m_hat = estimate[0]
sigma_m = np.sqrt(error[0][0])

print("\npython fit Least Squares estimator for mass " + str(m_hat))
print("python fit Least Squares uncertainty " + str(sigma_m))

plt.errorbar(F, a, yerr=sigma_a, fmt='o')

F_line = np.arange(0.8*min(F), 1.3*max(F), 1)
a_line = pdf(F_line, m_hat)
plt.plot(F_line, a_line, '--', color='red')
plt.xlabel('F [N]')
plt.ylabel(r'$a [ms^{-2}]$')

plt.savefig("Problem_1.pdf")
plt.clf()

# Analytical way

theta_hat_analitical = sum((np.array(a)*np.array(F))/np.array(sigma_a)**2)/sum(np.array(F)**2/np.array(sigma_a)**2)
sigma_theta_analitical = np.sqrt(1./(sum(np.array(F)**2/np.array(sigma_a)**2)))

print("\nAnalytical Least Squares estimator for theta " + str(theta_hat_analitical))
print("Analytical Least Squares uncertainty for theta " + str(sigma_theta_analitical))

mass_hat_analitical = 1./ theta_hat_analitical
sigma_mass_analitical = (1./ theta_hat_analitical**2 ) * sigma_theta_analitical

print("\nAnalytical Least Squares estimator for mass " + str(mass_hat_analitical))
print("Analytical Least Squares uncertainty for mass " + str(sigma_mass_analitical))

# Problem 2
def chi2(theta, a, F, sigma_a):
    return (sum(np.array(a)**2/np.array(sigma_a)**2)) - 2. * theta * sum((np.array(a)*np.array(F))/np.array(sigma_a)**2) + theta**2 * (sum(np.array(F)**2/np.array(sigma_a)**2))

x = np.linspace(9.0, 11.0, 100000)
chi2_line = chi2(x, a, F, sigma_a)
plt.plot(x, chi2_line)
plt.xlabel(r'$\theta [m^{-1}]$')
plt.ylabel(r'$\chi^2(\theta)$')

plt.savefig("Problem_2.pdf")

index = np.argmin(np.array(chi2_line))
theta_hat_chi2line = x[index]

def moved_chi2(x):
    return chi2(x, a, F, sigma_a) - chi2_line[index] - 1.0

sigma_down = theta_hat_chi2line - brentq(moved_chi2, 9.0, x[index])
sigma_up = brentq(moved_chi2, x[index], 11.0) - theta_hat_chi2line

print("\nLeast Squares estimator from the chi2 curve for theta " + str(theta_hat_chi2line))
print("Sigma_up uncertainty from the chi2 curve for theta = " + str(sigma_up))
print("Sigma_dn uncertainty from the chi2 curve for theta = " + str(sigma_down))

mass_hat_chi2line = 1./ theta_hat_chi2line
sigma_up_mass_chi2line = (1./ theta_hat_chi2line**2 ) * sigma_up
sigma_down_mass_chi2line = (1./ theta_hat_chi2line**2 ) * sigma_down

print("\nLeast Squares estimator from the chi2 curve for mass " + str(mass_hat_chi2line))
print("Sigma_up uncertainty from the chi2 curve for mass = " + str(sigma_up_mass_chi2line))
print("Sigma_down uncertainty from the chi2 curve for mass = " + str(sigma_down_mass_chi2line))

# Problem 3
plt.plot(x, chi2_line)
plt.axhline(y=chi2_line[index] + 1.0, color='r', linestyle='--')
plt.axhline(y=chi2_line[index] , color='r', linestyle='--')

plt.vlines(x=theta_hat_chi2line, ymin = 0., ymax=chi2_line[index], color='r', linestyle='--')
plt.vlines(x=theta_hat_chi2line + sigma_up , ymin = 0.,ymax=chi2_line[index] + 1.0, color='r', linestyle='--')
plt.vlines(x=theta_hat_chi2line - sigma_down , ymin = 0.,ymax=chi2_line[index] + 1.0, color='r', linestyle='--')

plt.xlabel(r'$\theta [m^{-1}]$')
plt.ylabel(r'$\chi^2(\theta)$')

plt.ylim([0., 11.])

plt.savefig("Problem_3.pdf")

