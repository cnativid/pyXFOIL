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
        self.solidity =N_b*scipy.integrate.quad(lambda x: self.c, 0, R)[0]/self.A
    
    
    def chord(self, r): # chord function of r
        return self.c

    def twist(self, r):
        rbar = r/self.R
        # return 0 
        # return np.amin([10.0, np.pi/180/(r/rotor.R)])
        return np.pi/180*((55-80*rbar)+(55*rbar-22)*(rbar > 0.4))

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
        # prandtl tip losses

        f_tip = self.N_b/2*(self.R-r)/(r*np.sin(inflow_angle))
        f_root = 1
        f = f_tip*f_root
        
        F = 2/np.pi*np.arccos(np.exp(-f))

        return self.N_b/2*rho*U**2*self.chord(r)*C_x*F
    
    def eval_dQ_BET(self, r, rpm, collective, inflow_angle):
        return self.eval_dT_BET(r, rpm, collective, inflow_angle)*r
    
    def eval_dP_BET(self, r, rpm, collective, inflow_angle):
        U = (r*rpm)/np.cos(inflow_angle)
        alpha = self.twist(r) + collective - inflow_angle
        C_t = Cl_lookup(alpha)*np.sin(inflow_angle) + Cd_lookup(alpha)*np.cos(inflow_angle)
        return self.N_b/2*rho*U**2*self.chord(r)*C_t*r*rpm
    
    def solve_T(self, rpm, collective, U_climb):
        return scipy.integrate.quad(lambda r: self.eval_dT_mom(r, rpm, self.solve_inflow_angle(r, rpm, collective, U_climb)), 0.2*self.R, self.R)

    def solve_P(self, rpm, collective, U_climb):
        # return scipy.integrate.quad(lambda r: self.eval_dP_BET(r, rpm, collective, self.solve_inflow_angle(r, rpm, collective, U_climb)), 0, self.R)
        return scipy.integrate.quad(lambda r: self.eval_dP_BET(r, rpm, collective, self.solve_inflow_angle(r, rpm, collective, U_climb)), 0, self.R)

    def solve_collective(self, rpm, T, U_climb):
        return scipy.optimize.fsolve(lambda collective: T - self.solve_T(rpm, collective, U_climb)[0], 0 )
        

    
def Cl_lookup(alpha):  # alpha is in rad
    return 5.73*alpha
    # return 2*np.pi*alpha

def Cd_lookup(alpha):  # alpha is in rad
    return 0.011
