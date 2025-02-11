import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import random
from scipy.optimize import brentq

# Problem 1
def Factorial( n ):
    factorial = 1;
    if ( n > 1 ):
        for i in range(1,n+1):
            factorial *= i
    return factorial

def Binomial( r, p, N):
    bin = 0.
    bin = Factorial(N)/(Factorial(r)*Factorial(N-r))*(p**r)*(1-p)**(N-r)
    return bin

# Problem 2
def ClopperPearson_u(r, N, CL, N_steps = 1000):
    step = 0.001
    binomial_sum = 0.
    p_upper = 0.
    p_temp = 0.

    for i_step in range(N_steps + 1):
        p_temp = i_step * step

        for i in range(r+1, N+1):
            binomial_sum += Binomial(i,p_temp,N)

        if ( binomial_sum >= (1-(1-CL)/2) ):
            p_upper = p_temp
            break

        elif (i_step == N_steps):
            p_upper = 1.00
        else:
            binomial_sum = 0

    return p_upper

def ClopperPearson_d(r, N, CL, N_steps = 1000):
    step = 0.001
    binomial_sum = 0.
    p_lower = 0.
    p_temp = 0.

    for i_step in range(N_steps + 1):
        p_temp = 1.0 - i_step * step

        for i in range(r):
            binomial_sum += Binomial(i,p_temp,N)

        if ( binomial_sum >= (1-(1-CL)/2) ):
            p_lower = p_temp
            break
        else:
            binomial_sum = 0

    return p_lower

def ClopperPearson(r, N, CL, N_steps=1000):
    p_u = ClopperPearson_u(r, N, CL, N_steps)
    p_d = ClopperPearson_d(r, N, CL, N_steps)
    return p_d, p_u

print("==============================================")
print("Clopper-Pearson intervals for N=10 and CL=0.68")
print("==============================================")
for i in range(0,10):
    print("[" + str(round(ClopperPearson(i, 10, 0.68)[0],2)) + "," + str(round(ClopperPearson(i, 10, 0.68)[1],2)) + "]")

# Problem 3
fig, ax = plt.subplots()

for i in range(0,11):
    p_d, p_u = ClopperPearson(i, 10, 0.68)
    x = np.linspace(i - 0.5,  i + 0.5)
    ax.fill_between(x, p_u + x * 0, x * 0 + p_d, color = "blue", alpha=0.3)

ax.set_xlim(-1, 11)
ax.set_ylim(0, 1)
ax.grid()

plt.xlabel('r')
plt.ylabel('p')

plt.title('Neyman Confidence Belt for N = 10')

plt.savefig("Problem_3.pdf")
plt.clf()

# Problem 4

p_true = 1./6. # true probability to roll a 6
n_covered = 0

for e in range(1000):
    n_rolled_sixes = 0
    for i in range(10): # throw a dice 10 times
        if( random.randint(1,6) == 6):
            n_rolled_sixes += 1 # count the number of 6
    p_d, p_u = ClopperPearson(n_rolled_sixes, 10, 0.68)
    if( p_true >= p_d and p_true <= p_u):
        n_covered += 1

print("\n============================================================================")
print("For the dice throwing experiment, the 1 sigma CP CL is actually a " + str(n_covered/10.) + "% CL!")
print("============================================================================")
