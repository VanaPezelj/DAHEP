import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

np.random.seed(45)
n_samples = 1000
true_mean = 5.0
true_sigma = 2.0
data = np.random.normal(true_mean, true_sigma, n_samples)

def logL(data, mean, sigma):
    return -0.5 * np.sum(((data - mean) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))

likelihood_ratios = []
for _ in range(10000):
    simulated_data = np.random.normal(true_mean, true_sigma, n_samples)
    # H0
    mean_0 = 5.0 # H0 is correct -> mean_0 = true_mean
    sigma_1 = np.std(simulated_data, ddof=1)
    logL_0 = logL(simulated_data, mean_0, sigma_1)
    # H1
    mean_1 = np.mean(simulated_data)
    sigma_1 = np.std(simulated_data, ddof=1)
    logL_1 = logL(simulated_data, mean_1, sigma_1)

    lr = -2 * (logL_0 - logL_1)
    likelihood_ratios.append(lr)

plt.hist(likelihood_ratios, bins=100, density=True, alpha=0.7, label="Simulated")
x = np.linspace(0, 7 , 100)
plt.plot(x, chi2.pdf(x, 1), label=r"$\chi^2_1$", color="red")
plt.xlabel("Likelihood Ratio")
plt.legend()
plt.savefig("Problem_1.pdf")