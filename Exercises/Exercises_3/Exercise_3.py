import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import random


def Gaussian(x, mu, sigma):
    return (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))


# Problem 1
x = np.linspace(-5,5,1000)
f1 = Gaussian(x,0.0,1.0)
f2 = Gaussian(x,1.0,1.0)
f3 = Gaussian(x,0.0,2.0)


plt.plot(x, f1, label = "mu = 0, sigma = 1")
plt.plot(x, f2, label = "mu = 1, sigma = 1")
plt.plot(x, f3, label = "mu = 0, sigma = 2")
plt.legend(loc="upper left")


plt.savefig('Problem_1.pdf')
plt.clf()


print("Probability to produce the new particle with a mass of 205 GeV or more is " + str(100*(integrate.quad(lambda x: Gaussian(x,200.0, 2.0), 205.0, +np.inf))[0]) + "%.")
print("Probability to produce the new particle with a mass with mass between 199 and 201 GeV " + str(100*(integrate.quad(lambda x: Gaussian(x,200.0, 2.0), 199.0, 201.0))[0]) + "%.")
print("Probability to produce the two particles with masses above 203 GeV is " + str(100*(integrate.quad(lambda x: Gaussian(x,200.0, 2.0), 203.0, +np.inf))[0]**2) + "%.")




# Problem 2
def CDF(pdf, x_min = -5.0, x_max = 5.0, steps = 1000):
    cdf = []
    for i in np.linspace(x_min,x_max,steps):
        cdf.append((integrate.quad(lambda x: pdf(x), -np.inf, i))[0])
    return cdf


def pdf_1(x):
    return Gaussian(x, 0.0, 1.0)


cdf = CDF(pdf_1)
plt.plot(x, cdf, label = "CDF(x)")
plt.legend(loc="upper left")
plt.savefig('Problem_2.pdf')
plt.clf()


# Problem 3 - 4
def random_pdf(pdf, N, x_min, x_max):
    x_rand = []
    n_tries = 0
    while (len(x_rand) < N):
        n_tries += 2
        (x,y)= (random.uniform(x_min,x_max), random.uniform(0.0,1.0/np.sqrt(2*np.pi)))
        if(pdf(x) > y):
            x_rand.append(x)
    print("I had to generate " + str(n_tries) + " random numbers to draw " + str(N) + " random numbers according to a given PDF.")
    return x_rand


N = 10000
x_rand = random_pdf(pdf_1, N, -5.0, 5.0)
plt.title("Acceptance-rejection method")
plt.hist(x_rand, bins = 20, label="MC toys")
plt.savefig('Problem_34.pdf')
plt.clf()


# Problem 5
def random_pdf_fast(pdf, N, x_min, x_max):
    x_rand = []
    x = np.linspace(x_min,x_max,1000)
    cdf = CDF(pdf, x_min, x_max)
    n_tries = 0
    while (len(x_rand) < N):
        n_tries += 1
        u = random.random()
        index = np.argmin(np.abs(np.array(cdf)-u))
        x_rand.append(x[index])
    print("I had to generate " + str(n_tries) + " random numbers to draw " + str(N) + " random numbers according to a given PDF.")
    return x_rand


x_rand = random_pdf_fast(pdf_1, 10000, -5.0, 5.0)


plt.title("Inversion method")
plt.hist(x_rand, bins = 20, label="MC toys")
plt.savefig('Problem_5.pdf')
