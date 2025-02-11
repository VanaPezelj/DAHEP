import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import uproot

data = uproot.open("/home/public/data/GaussData.root:tree")["x_observed"]

def logL(data, mu, sigma):
    return -0.5 * np.sum(((data - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))

def likelihood_ratio(mu, data):
    mu_1 = np.mean(data)
    sigma_1 = np.std(data,ddof=1)
    logL_0 = logL(data,mu, sigma_1)
    logL_1 = logL(data, mu_1, sigma_1)
    return -2 * (logL_0 - logL_1)

mu_values = np.linspace(4.5, 5.5, 1000)  
lr_values = np.array([likelihood_ratio(mu, data) for mu in mu_values])

CL95p4 = 4 #95.4% CI for ndof=1

inside = np.where(lr_values <= CL95p4)[0]

lower_bound = mu_values[inside[0]]
upper_bound = mu_values[inside[-1]]


plt.plot(mu_values, lr_values, label='Likelihood Ratio')
plt.axhline(y=CL95p4, color='r', linestyle='--', label='95.4% CL')
if lower_bound and upper_bound:
    plt.axvline(x=lower_bound, color='g', linestyle='--', label=f'Lower Bound: {lower_bound:.2f}')
    plt.axvline(x=upper_bound, color='g', linestyle='--', label=f'Upper Bound: {upper_bound:.2f}')

plt.xlim(4.5, 5.5)

plt.ylim(0, 5)

plt.xlabel(r'$\mu$')
plt.ylabel('Likelihood ratio')
plt.title('Likelihood Ratio Test with 95.4% Confidence Interval')
plt.legend(loc='lower right')
plt.savefig("Problem_3.pdf")

print(f'95.4% Confidence Interval for mu: ({lower_bound:.2f}, {upper_bound:.2f})')
