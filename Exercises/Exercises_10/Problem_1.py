import uproot
import matplotlib.pyplot as plt
import numpy as np
from numba_stats import crystalball, bernstein
from iminuit import Minuit, cost

def TotalCDF(bin_edges,s,mu,sigma,beta, m,b,a0,a1,a2):

    cb_CDF = crystalball.cdf(bin_edges,beta,m,mu,sigma)
    bernstein_CDF = bernstein.integral(bin_edges,[a0,a1,a2],60,120)
    return (s * cb_CDF+ b * bernstein_CDF)
    # for more precise result remove b from fit with b = len(data)-s

def MakeFit(data):

    counts, bin_edges = np.histogram(data, bins=60, range=[60,120])
    counts = np.asarray(counts)
    bin_edges = np.asarray(bin_edges)
    c = cost.ExtendedBinnedNLL(counts, bin_edges, TotalCDF)
    m = Minuit(c, s=len(data)/2, mu=90, sigma=7,beta=1,m=20, b=10000, a0=1, a1=1, a2=1)
    m.limits["s"] = (0, len(data))
    m.limits["sigma"] = (0, 30)
    m.limits["mu"] = (80, 100)
    m.limits["beta"] = (0, 5)
    m.limits["m"] = (1,30)
    m.limits["b"] = (0, None)
    m.limits["a0"] = (0.,3)
    m.limits["a1"] = (0.,3)
    m.limits["a2"] = (0.,3)
    m.migrad()
    print(m.values)
    return m

def MakePlot(m,data,num,name):

    x = np.linspace(60,120,1000)
    y_total = m.values["s"] * crystalball.pdf(x,m.values["beta"],m.values["m"],m.values["mu"],m.values["sigma"]) + m.values["b"] * bernstein.density(x, [m.values["a0"],m.values["a1"],m.values["a2"]], 60, 120)
    y_bkg = m.values["b"] * bernstein.density(x, [m.values["a0"],m.values["a1"],m.values["a2"]], 60, 120)
    plt.figure(num)
    plt.hist(data,bins=60,range=[60,120],label="data")
    plt.plot(x,y_total,label="CB+Bernstein")
    plt.plot(x,y_bkg,label="Bernstein")
    plt.title(f"{name}")
    plt.xlabel(r"$m_{e^+e^-}$/GeV")
    plt.legend()
    plt.savefig(f"{name}.pdf")

def EffCalc(m_pass,m_fail):
    
    N_pass  = m_pass.values["s"]
    N_fail  = m_fail.values["s"]
    error_pass = m_pass.errors["s"]
    error_fail = m_fail.errors["s"]
    
    N_total = N_pass + N_fail
    Eff = N_pass/N_total

    dEff_dpass = N_fail / (N_total ** 2)
    dEff_dfail = -N_pass / (N_total ** 2)
    StatUnc = np.sqrt((dEff_dpass * error_pass) ** 2 + (dEff_dfail * error_fail) ** 2)

    return Eff, StatUnc

tree = uproot.open("/home/public/data/TagAndProbe/Run2022E_merged.root:tnpEleIDs")["fitter_tree"]

PassID_pairs = tree.arrays(["pair_mass"],"(tag_Ele_pt>50) & (tag_Ele_abseta<2.17) & (passingMVASummer18ULwpHZZ==1) & (pair_mass>60) & (pair_mass<120) & (el_q*tag_Ele_q==-1)")
data_pass = PassID_pairs["pair_mass"]

FailID_pairs = tree.arrays(["pair_mass"],"(tag_Ele_pt>50) & (tag_Ele_abseta<2.17) & (passingMVASummer18ULwpHZZ!=1) & (pair_mass>60) & (pair_mass<120) & (el_q*tag_Ele_q==-1)")
data_fail = FailID_pairs["pair_mass"]

m_pass = MakeFit(data_pass)
m_fail = MakeFit(data_fail)

MakePlot(m_pass,data_pass,1,"PassingProbes")
MakePlot(m_fail,data_fail,2,"FaillingProbes")

eff = EffCalc(m_pass,m_fail)

print(f"eff={eff[0]}±{eff[1]} statistical")