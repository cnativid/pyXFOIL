import numpy as np
import scipy
import matplotlib.pyplot as plt
from BEMT import Rotor

rho = 1
U_climb = 0

n15_alpha, n15_Cl, n15_Cd = np.loadtxt("/home/canativi/Documents/Python/BEMT/test/data/naca0012_polar.dat", skiprows = 12, unpack= True, usecols = (0,1,2))

def Cl_lookup(alpha):
    alpha_deg = alpha*180/np.pi
    return (1-2*(alpha<0))*np.interp(np.abs(alpha_deg), n15_alpha, n15_Cl)
    
def Cd_lookup(alpha):
    alpha_deg = alpha*180/np.pi
    return (1-2*(alpha<0))*np.interp(np.abs(alpha_deg), n15_alpha, n15_Cd)
    
if __name__ == "__main__":    
    # R = 2.5 # ft
    # R_c = 1.5/12 # ft
    c = 2/12 # ft
    rpm = 960
    collectives = np.linspace(0,15,16)
    r_stations = 1.5/12 + (2.5-1.5/12)*(1-np.cos(np.pi*np.linspace(0,1,101)))/2
    
    f1, ax_ct= plt.subplots()
    f2, ax_cp= plt.subplots()
    for N_b in [2,3,4,5]:
        rotor = Rotor(c=c, R=2.5, R_c=1.5/12, N_b=N_b, 
                 r_stations = r_stations,
                 rpm = rpm, collective = 0,
                 Cl_lookup = Cl_lookup, Cd_lookup = Cd_lookup)
        
        C_T_array = []
        C_P_array = []
        
        for collective in collectives:
            rotor.collective = collective*np.pi/180
            inflow_angles, alpha, Cl, Cd = rotor.solve_BEMT(U_climb)
            
            
            C_T = rotor.eval_T_BET(r_stations, inflow_angles, U_climb)/(rho*rotor.A*(rpm*rotor.R)**2)
            C_P = rotor.eval_P_BET(r_stations, inflow_angles, U_climb)/(rho*rotor.A*(rpm*rotor.R)**3)
            C_T_array.append(C_T)
            C_P_array.append(C_P)
            
        ax_ct.plot(collectives, C_T_array, label = f"B = {N_b:g} | BEMT")
        try:
            col_exp, ct_exp = np.loadtxt(f"validation_data/KH_{N_b:g}.dat", unpack = True)
            ax_ct.scatter(col_exp, ct_exp/2, label = f"B = {N_b:g} | EXP")
        except:
            pass
        ax_ct.set_xlabel("Collective [deg]")
        ax_ct.set_ylabel("C_T")
        
        try:
            ax_cp.plot(collectives, C_P_array, label = f"B = {N_b:g} | BEMT")
            ax_cp.scatter(*np.loadtxt(f"validation_data/KH_P{N_b:g}.dat", unpack = True), label = f"B = {N_b:g} | EXP")
        except:
            pass
        
        ax_cp.set_xlabel("Collective [deg]")
        ax_cp.set_ylabel("C_P")
    ax_ct.legend()
    ax_cp.legend()
    f1.savefig("ct.png")
    f2.savefig("cp.png")
    plt.show()
  
    