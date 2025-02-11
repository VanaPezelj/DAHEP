import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

def CalculateInvMass(particle):
    return np.sqrt(pow(particle["E"],2)-pow(particle["px"],2)-pow(particle["py"],2)-pow(particle["pz"],2))


file = uproot.open("/home/public/data/ggH125/ZZ4lAnalysis.root")
tree = file["ZZTree"]["candTree"]

LepArrays=tree.arrays(["LepPt","LepEta","LepPhi"])

LepMomentumCartesian = {
    "px":LepArrays["LepPt"] * np.cos(LepArrays["LepPhi"]),
    "py":LepArrays["LepPt"] * np.sin(LepArrays["LepPhi"]),
    "pz":LepArrays["LepPt"] * np.sinh(LepArrays["LepEta"]),
}


E = sum([pow(value,2) for value in LepMomentumCartesian.values()]) #E=px^2+py^2+pz^2
E=np.sqrt(E)

Lep4MomentumCartesian = {**LepMomentumCartesian, "E": E} #add energy to create 4-momentum vector

Z1,H={},{}
for key in Lep4MomentumCartesian.keys(): #need to convert to akk array to apply arithmetic operations
    Z1[str(key)]=ak.Array([sum(i[:2]) for i in Lep4MomentumCartesian[str(key)]]) #Z1=l1+l2
    H[str(key)]=ak.Array([sum(i[:4]) for i in Lep4MomentumCartesian[str(key)]]) #H=l1+l2+l3+l4

Z1_InvMass, H_InvMass = CalculateInvMass(Z1), CalculateInvMass(H)

plt.figure(figsize=(8, 6))
plt.hist(Z1_InvMass, bins=100, range=(50,150), color='blue', alpha=0.7,label="Z1")
plt.hist(H_InvMass, bins=100, range=(50,150), color='red', alpha=0.7, label="H")
plt.xlabel("$m_{4l}$/GeV")
plt.ylabel("Events")
plt.title("ggH125")
plt.legend()
plt.savefig('ggH125_InvMass.png')
