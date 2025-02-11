import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv

np.random.seed(42)

def GenerateHypothesis(mu,sigma):
    AvgGenHeight_H = []
    for i in range(10**6):
        samples = np.random.normal(mu, sigma, size=100)
        AvgGenHeight_H.append(np.average(samples))
    return AvgGenHeight_H

Height = uproot.open("/home/public/data/Height/Height.root")["Tree"]["height"]
AvgHeightObs = np.average(Height)

HeightDict={
    "Spain" : [168,7],
    "France" : [165.5,7.1],
    "Italy" : [166.1,6.5],
    "Netherlands": [170.3,7.5]
}

for country, params in HeightDict.items():
    mean, std_dev = params  
    HeightDict[country].append(GenerateHypothesis(mean,std_dev))

H_Spain, H_France, H_Italy, H_Netherlands = np.array(HeightDict["Spain"][2]), np.array(HeightDict["France"][2]), np.array(HeightDict["Italy"][2]), np.array(HeightDict["Netherlands"][2])

#Spain VS France
plt.figure(1)
plt.hist(H_Spain,bins=100,label='Spain',alpha=0.7,density=True)
plt.hist(H_France,bins=100,label='France',alpha=0.7,density=True)
plt.axvline(x = AvgHeightObs , color = 'r', label = 'observed')
plt.xlabel(r"$\overline{h}$[cm]")
plt.legend()
plt.savefig("SpainVsFrance.pdf")
p_SF = 1 - len(H_France[H_France>AvgHeightObs])/len(H_Spain[H_Spain<AvgHeightObs]) #H1 and H0 total lengths cancel out
print(f"Spain VS France: Rejecting HO with {p_SF*100}% CL") 

#Spain VS Italy
plt.figure(2)
plt.hist(H_Spain,bins=100,label='Spain',alpha=0.7,density=True)
plt.hist(H_Italy,bins=100,label='Italy',alpha=0.7,density=True)
plt.axvline(x = AvgHeightObs , color = 'r', label = 'observed')
plt.xlabel(r"$\overline{h}$[cm]")
plt.legend()
plt.savefig("SpainVsItaly.pdf")
p_SI = 1 - len(H_Italy[H_Italy>AvgHeightObs])/len(H_Spain[H_Spain<AvgHeightObs])
print(f"Spain VS Italy: Rejecting HO with {p_SI*100}% CL") 

#Spain VS Netherlands
plt.figure(3)
plt.hist(H_Spain,bins=100,label='Spain',alpha=0.7,density=True)
plt.hist(H_Netherlands,bins=100,label='Netherlands',alpha=0.7,density=True)
plt.axvline(x = AvgHeightObs , color = 'r', label = 'observed')
plt.xlabel(r"$\overline{h}$[cm]")
plt.legend()
plt.savefig("SpainVsNetherlands.pdf")
p_SN = 1 - len(H_Netherlands[H_Netherlands<AvgHeightObs])/len(H_Spain[H_Spain>AvgHeightObs])
print(f"Spain VS Netherlands: Rejecting HO with {p_SN*100}% CL")