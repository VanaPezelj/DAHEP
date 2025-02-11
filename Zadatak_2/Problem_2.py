import uproot
import numpy as np
from iminuit import cost, Minuit
from numba_stats import truncexpon, truncnorm
import matplotlib.pyplot as plt

def BkgPDF(x,b,tau):
    return b * truncexpon.pdf(x, 0, 10, 0, tau)

def TotalPDF(x, s, b, mu, sigma, tau):
    return s + b, s * truncnorm.pdf(x, 0, 10, mu, sigma) + b * truncexpon.pdf(x, 0, 10, 0, tau)
 
x = uproot.open("/home/public/data/SigBkg_Mix.root")["tree"]["InvMass"]
x=np.array(x)

c = cost.ExtendedUnbinnedNLL(x, TotalPDF)

m = Minuit(c, s=5000, b=5000, mu=8, sigma=1, tau=1)
m.limits["s", "b", "sigma", "tau"] = (0, None)
m.migrad()

print(f"s={m.values['s']}±{m.errors['s']}")
print(f"b={m.values['b']}±{m.errors['b']}")

plt.hist(x,bins=100,range=[0,10],label="data",density=True)

x = np.linspace(0,10,1000)
y_tot = TotalPDF(x,m.values["s"], m.values["b"],m.values["mu"],m.values["sigma"],m.values["tau"])[1]/(m.values["s"]+m.values["b"])
y_bkg = BkgPDF(x,m.values["b"],m.values["tau"])/(m.values["s"]+m.values["b"])
plt.plot(x, y_tot,label="Gauss+Exp")
plt.plot(x, y_bkg,label="Exp")
plt.xlabel("Invariant Mass / GeV")
plt.legend()
plt.savefig("Problem_2.pdf")