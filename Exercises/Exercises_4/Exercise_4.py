import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import random
import uproot
from scipy.optimize import brentq
from scipy import stats
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

# Problem 1
tree = uproot.open("/home/public/data/Lifetime/Lifetime.root:Tree")
t = tree["t"].array()

plt.hist(t, bins = 50)
plt.xlabel('t [s]')
plt.ylabel('N')
plt.savefig("Problem_1.pdf")
plt.clf()


# Problem 2
def decay_pdf(t,tau):
    return (1./tau)*np.exp(-1.0*t/tau)

x = np.linspace(0, 10., 1000)
f1 = decay_pdf(x, 1.0)

f2 = decay_pdf(x, 2.0)
f3 = decay_pdf(x, 0.5)

plt.plot(x, f1, label="τ = 1.0")
plt.plot(x, f2, label="τ = 2.0")
plt.plot(x, f3, label="τ = 0.5")
plt.xlabel('t [s]')
plt.ylabel('N')
plt.legend(loc="upper right")
plt.savefig("Problem_2.pdf")
plt.clf()

print(f"Probability is {100*(integrate.quad(lambda x: decay_pdf(x, 2.0), 0.0, 1.0))[0]} %")

# Problem 3
x = np.linspace(0., 3., 1000)
l = decay_pdf(1.0, x)
plt.plot(x, l)
plt.xlabel('τ [s]')
plt.ylabel('L(τ)')
plt.savefig("Problem_3.pdf")
plt.clf()

# Problem 4
def lnL(tau, N, sum_t):
    return 2.0*N*np.log(tau) + 2.0*sum_t/tau

x = np.linspace(1.0, 1.5, 1000)
logLikelihood = lnL(x, len(t), sum(t))
plt.plot(x, logLikelihood)
plt.xlabel('τ [s]')
plt.ylabel('-2lnL(τ)')
plt.savefig("Problem_4.pdf")
plt.clf()

index = np.argmin(np.array(logLikelihood))
tau_hat = x[index]
print("Exact tau = ", np.mean(t))
print("Maximum likelihood estimator for mean lifetime is " + str(tau_hat))

def moved_lnL(tau):
    return lnL(tau, len(t), sum(t)) - (logLikelihood[index] + 1.0)

sigma_down = tau_hat - brentq(moved_lnL, 1.0, x[index])
sigma_up = brentq(moved_lnL, x[index], 1.5) - tau_hat

print("Sigma_up uncertainty = " + str(sigma_up))
print("Sigma_dn uncertainty = " + str(sigma_down))

#Problem 5
cost = UnbinnedNLL(t.to_numpy(), decay_pdf)
m = Minuit(cost, tau=1)
m.limits["tau"] = (0.5, 5)
m.migrad()
m.hesse()
print(f"Minuit estimation:tau={m.values['tau']}±{m.errors['tau']}")

