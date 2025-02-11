import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Distribution of Sums
np.random.seed(42)
SumArray=[]
i, j, k = np.linspace(0.1,3,100), np.linspace(0,100,100), np.linspace(1,100,100)
for iter in range(0,10000):
   ExpSamples = np.random.exponential(i)
   GaussSamples = np.random.normal(j,5)
   PoissonSamples = np.random.poisson(k)
   Sum = np.sum(ExpSamples)+np.sum(GaussSamples)+np.sum(PoissonSamples)
   SumArray.append(Sum)

plt.hist(SumArray, bins=50, color='skyblue',density=True,label="Generated Histogram")

#CLT prediction

#Exp: Mean=lambda, Var=lambda**2, Gauss: Mean=mu, Var=sigma**2=25, Poisson: Mean=Lambda, Var = Lambda  
mean = np.sum(i)+np.sum(j)+np.sum(k)
var = np.sum(i**2)+5**2*np.size(j)+np.sum(k)

print(f"CLT predicts that sum is distributed as Gauss({mean},{np.sqrt(var)})")

x = np.linspace(9900,10500,10000)
plt.plot(x, norm.pdf(x,mean,np.sqrt(var)),label="CLT Prediction")
plt.legend()
plt.xlabel("Sum")
print(mean,np.sqrt(var))

plt.savefig("CLT.pdf")



