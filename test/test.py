import numpy as np
import scipy
import matplotlib.pyplot as plt


inflow_c = 0

def CL_lookup(alpha):
    # return (np.pi**2)*(alpha)/90
    return 5.73*alpha*np.pi/180

def solve_inflow(solidity, r, collective_deg):
    if r == 0:
        return 0
    else:
        inflow = 0
        res = 1
        while res > 1e-5:
            res = solidity/2*CL_lookup(collective_deg-180/np.pi*inflow/r)*(r**2) - 4*inflow*(inflow-inflow_c)*r
            inflow = inflow + res/10
        return inflow

def eval_dC_T(solidity, r, collective_deg):
    inflow = solve_inflow(solidity, r, collective_deg)
    return 4*inflow*(inflow-inflow_c)*r

def eval_C_T(collective_deg, solidity):
    r_array = np.linspace(0,1,141)
    ans = scipy.integrate.trapezoid([eval_dC_T(solidity, r, collective_deg) for r in r_array], x = r_array)
    print(ans)
    return ans
        
def solve_rotor(C_T, solidity):
    collective_deg = 0
    res = 1
    while res > 1e-7:
        res = C_T - eval_C_T(collective_deg, solidity)
        print(collective_deg,res)
        collective_deg += 1000*res
    return collective_deg

C_T = 0.02/2
solidity = 0.10610329539459687
collective_deg = solve_rotor(C_T, solidity)
r_array = np.linspace(0,1,141)
inflow = [solve_inflow(solidity, r, collective_deg) for r in r_array]
plt.plot(r_array, inflow)
plt.plot(r_array, solidity*2*np.pi/16*(np.sqrt(1+32*collective_deg*np.pi/180*r_array/(solidity*2*np.pi))-1))
plt.xlabel("r/R")
plt.xlabel("inflow ratio")
plt.title("C_T = 0.008, solidity = 0.1")
plt.savefig("test.png")
print(collective_deg)
plt.show()
    