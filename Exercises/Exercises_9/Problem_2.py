import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

ndof = [1, 2, 3, 4, 5]
x = np.linspace(0, 15, 1000)

plt.figure(figsize=(8, 6))
for dof in ndof:
    cdf = chi2.cdf(x, dof)
    plt.plot(x, cdf, label=f'dof = {dof}')
    if(dof==1):
        index = np.where(cdf>=0.954)[0][0] # np where returns index as an array
        threshold = x[index]

print(f'95.4% CL for ndof=1 : {threshold}')        
plt.title('CDF of Chi-Squared Distribution for Different Degrees of Freedom')
plt.xlabel('x')
plt.ylabel('CDF')
plt.legend(title='Degrees of Freedom')

plt.savefig("Problem_2.pdf")