import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv

np.random.seed(42)
Height = uproot.open("/home/public/data/Height/Height.root")["Tree"]["height"]
AvgHeightObs = np.average(Height)

print(f"Observed Height:{AvgHeightObs}")

AvgGenHeight_H0 = []
for i in range(10**6):
    samples = np.random.normal(165.5, 7.1, size=100)
    AvgGenHeight_H0.append(np.average(samples))

counts,bins, _ = plt.hist(AvgGenHeight_H0,bins=100,density=True,label='$H_0$',range=[160,170])
plt.axvline(x = AvgHeightObs , color = 'r', label = 'observed')
plt.legend()
plt.savefig("Problem_1.pdf")


AvgGenHeight_H0 = np.array(AvgGenHeight_H0)
p_value = len(AvgGenHeight_H0[AvgGenHeight_H0 >= AvgHeightObs]) / len(AvgGenHeight_H0)
z_score = np.sqrt(2) * erfcinv(2 * p_value)
print(f"p value={p_value}, z_score={z_score}")
