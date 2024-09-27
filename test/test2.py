import numpy as np
import scipy
import matplotlib.pyplot as plt

rho = 1
U_climb = 0
class Rotor:
    def __init__(self, c, R, N_b):
        self.c = c
        self.R = R
        self.N_b = N_b
        self.A = np.pi*R**2
        self.solidity = (N_b*c*R)/self.A
    
    
    def chord(self, r): # chord function of r
        return self.c

    def twist(self, r):
        return 0 
        # return np.amin([10.0, np.pi/180/(r/rotor.R)])

    def solve_inflow_angle(self, r, rpm, collective, U_climb):
        return scipy.optimize.fsolve(self.res_inflow_angle, 0, args = (r, rpm, collective, U_climb))

    def res_inflow_angle(self, inflow_angle, r, rpm, collective, U_climb):
        return self.eval_dT_mom(r, rpm, inflow_angle) - self.eval_dT_BET(r, rpm, collective, inflow_angle)
    
    def eval_dT_mom(self, r, rpm, inflow_angle):
        U_p = (r*rpm)*np.tan(inflow_angle)
        return 4*np.pi*U_p*(U_p-U_climb)*r
    
    def eval_dT_BET(self, r, rpm, collective, inflow_angle):
        U = (r*rpm)/np.cos(inflow_angle)
        alpha = self.twist(r) + collective - inflow_angle
        C_x = Cl_lookup(alpha)*np.cos(inflow_angle) - Cd_lookup(alpha)*np.sin(inflow_angle)
        return self.N_b/2*rho*U**2*self.chord(r)*C_x
    
    def eval_T(self, rpm, collective, U_climb):
        return scipy.integrate.quad(lambda r: self.eval_dT_mom(r, rpm, self.solve_inflow_angle(r, rpm, collective, U_climb)), 0, self.R)

def Cl_lookup(alpha):  # alpha is in rad
    return 5.73*alpha
    # return 2*np.pi*alpha

def Cd_lookup(alpha):  # alpha is in rad
    return 0.011

    
if __name__ == "__main__":    
    R = 2.5 # ft
    c = 2/12 # ft
    rpm = 960
    collectives = np.linspace(0,15,16)
    
    plt.figure()
    for N_b in [2,3,4,5]:
        rotor = Rotor(c, R, N_b)
        print(rotor.solidity)
        C_T_array = []
        for collective in np.linspace(0,15,16):
            T = rotor.eval_T(rpm,collective*np.pi/180,U_climb)[0]
            C_T = T/(rho*rotor.A*(rpm*rotor.R)**2)
            C_T_array.append(2*C_T)
        plt.plot(collectives, C_T_array)
        try:
            plt.scatter(*np.loadtxt(f"validation_data/KH_{N_b:g}.dat", unpack = True))
        except:
            pass
    plt.figure()
    N_b = 5
    collective = 12*np.pi/180
    rotor = Rotor(c, R, N_b)
    T = rotor.eval_T(rpm,collective,U_climb)[0]
    C_T = 2*T/(rho*rotor.A*(rpm*rotor.R)**2)
    
    r_array = np.linspace(0,rotor.R,50)
    
    inflow_angles = np.array([rotor.solve_inflow_angle(r, rpm, collective, U_climb)[0] for r in r_array])
    
    inflow_ratio = r_array/rotor.R*inflow_angles
    plt.plot(r_array, inflow_ratio)
    plt.plot(r_array, rotor.solidity*2*np.pi/16*(np.sqrt(1+32*collective*(r_array/rotor.R)/(rotor.solidity*2*np.pi))-1))
    plt.show()