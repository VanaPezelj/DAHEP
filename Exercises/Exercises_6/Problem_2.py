import uproot
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

def Get_Dkin(path,c):
    dir = uproot.open(path)
    tree = dir["candTree"]
    p_ggH125, p_ggZZ = tree["p_GG_SIG_ghg2_1_ghz1_1_JHUGen"].array(), tree["p_QQB_BKG_MCFM"].array()
    Counters40, xsec, overallEventWeight = dir["Counters"].values()[39], tree["xsec"].array(), tree["overallEventWeight"].array()
    EventWeight = 137.0 * 1000 * xsec * overallEventWeight / Counters40
    Dkin = 1/(1+c*p_ggZZ/p_ggH125)
    return Dkin, EventWeight

D_kin_ggH125, EventWeights_ggH125 = Get_Dkin("/home/public/data/ggH125/ZZ4lAnalysis.root:ZZTree",1)
D_kin_qqZZ, EventWeights_qqZZ = Get_Dkin("/home/public/data/qqZZ/ZZ4lAnalysis.root:ZZTree",70)

plt.figure(1)

counts_ggH125, bins_ggH125, _ = plt.hist(D_kin_ggH125, bins=100, color='blue',density=True,label="ggH125", weights = EventWeights_ggH125, alpha=0.7, range=[0,1])
counts_qqZZ, bins_qqZZ, _ = plt.hist(D_kin_qqZZ, bins=100, color='red',density=True,label="qqZZ", weights = EventWeights_qqZZ, alpha=0.7, range=[0,1])

plt.xlabel("$D_{kin}^{bkg}$")
plt.legend()

plt.savefig("KinematicDiscriminant.pdf")

#ROC curve by hist integration

plt.figure(2)

bin_width=bins_ggH125[1]-bins_ggH125[0]
SigEff, BkgEff = [], []
for index, _ in enumerate(bins_ggH125):
    SigEff.append(np.sum(counts_ggH125[index:])*bin_width)
    BkgEff.append(np.sum(counts_qqZZ[index:])*bin_width)

plt.ylim(0.95,1.002)
plt.xlabel("Background Efficiency")
plt.ylabel("Signal Efficiency")
plt.plot(BkgEff,SigEff)
plt.savefig("ROC_binned.pdf")

area = simpson(SigEff, dx=bin_width)
print(f"AUC (binned) = {area}")

#ROC curve unbinned

plt.figure(3)

SigEff, BkgEff = [], []    
threshold=np.linspace(0,1,1001)
for thr in threshold:
    SigEff.append(np.sum(EventWeights_ggH125[D_kin_ggH125>thr])/np.sum(EventWeights_ggH125))
    BkgEff.append(np.sum(EventWeights_qqZZ[D_kin_qqZZ>thr])/np.sum(EventWeights_qqZZ))

plt.ylim(0.95,1.002)
plt.xlabel("Background Efficiency")
plt.ylabel("Signal Efficiency")
plt.plot(BkgEff,SigEff)
plt.savefig("ROC_unbinned.pdf")

area = simpson(SigEff, dx=1/(len(threshold)-1))
print(f"AUC (unbinned) = {area}")



