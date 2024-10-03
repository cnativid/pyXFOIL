import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import njit

rho = 1
U_climb = 0
class Rotor:
    def __init__(self, c=2/12, twist=0, R=2.5, R_c=1.5/12, N_b=2, 
                 r_stations = 1.5/12 + (2.5-1.5/12)*(1-np.cos(np.pi*np.linspace(0,1,101)))/2,
                 rpm = 960, collective = 0, 
                 Cl_lookup = None, Cd_lookup = None):
        
        if not callable(c):
            def chord_function(r):
                return c + r*0
            self.chord = chord_function
        else:
            self.chord = c

        if not callable(twist):
            def twist_function(r):
                return twist + r*0
            self.twist = twist_function
        else:
            self.twist = twist
    
        
        self.R = R
        self.R_c = R_c
        self.r_stations = r_stations
        self.Cl_lookup = Cl_lookup
        self.Cd_lookup = Cd_lookup
        self.collective = collective
        self.rpm = rpm
        self.N_b = N_b
        self.A = np.pi*R**2
        self.solidity = N_b*scipy.integrate.quad(lambda r: self.chord(r), 0, R)[0]/self.A
        
    def solve_BEMT(self, U_climb, inflow_angle0 = 0):
        res_inflow_angle = self.res_inflow_angle
        
        r_stations = self.r_stations
        inflow_angles_a = np.zeros(len(r_stations))
        inflow_angles_b = 0.9*np.pi/2*np.ones(len(r_stations))
        
        # initialize modified RF:
        res_a = res_inflow_angle(r_stations, inflow_angles_a, U_climb)
        res_b = res_inflow_angle(r_stations, inflow_angles_b, U_climb)
        res_c = np.ones(len(r_stations))
        while any(np.abs(res_c) > 1e-6):
            # inflow_angles_c = (inflow_angles_a*res_b - inflow_angles_b*res_a)/(res_b-res_a) # regular RF (slow)
            inflow_angles_c = inflow_angles_a + ( -res_a / ((res_b-res_a)/(inflow_angles_b-inflow_angles_a)**2))**0.5 # RF w/ vertex-point parabolic fit
            res_c = res_inflow_angle(r_stations, inflow_angles_c, U_climb)
            
            replace_b = res_a*res_c < 0
            replace_a = ~replace_b
            
            inflow_angles_b[replace_b] = inflow_angles_c[replace_b]
            res_b[replace_b] = res_c[replace_b]
            inflow_angles_a[replace_a] = inflow_angles_c[replace_a]
            res_a[replace_a] = res_c[replace_a]
        
        dT, alpha, Cl, Cd = self.eval_dT_BET(r_stations, inflow_angles_c, U_climb)
        
        # dP = self.eval_dP_BET(r_stations, self.rpm, self.collective, inflow_angles_c)
        return inflow_angles_c, alpha, Cl, Cd
    
    def eval_T_BET(self, r_stations, inflow_angles, U_climb):
        return np.trapz(self.eval_dT_BET(r_stations, inflow_angles, U_climb)[0], x = r_stations)
        
    def eval_P_BET(self, r_stations, inflow_angles, U_climb):
        return np.trapz(self.eval_dP_BET(r_stations, inflow_angles, U_climb), x = r_stations)


    def solve_inflow_angle(self, r, rpm, collective, U_climb):
        return scipy.optimize.fsolve(self.res_inflow_angle, 0, args = (r, rpm, collective, U_climb))

    def res_inflow_angle(self, r, inflow_angle, U_climb):
        return self.eval_dT_mom(r, inflow_angle, U_climb) - self.eval_dT_BET(r, inflow_angle, U_climb)[0]
    
    def eval_dT_mom(self, r, inflow_angle, U_climb):
        U_p = (r*self.rpm)*np.tan(inflow_angle)
        return 4*np.pi*U_p*(U_p-U_climb)*r
    
    def eval_dT_BET(self, r, inflow_angle, U_climb):
        # prandtl tip losses

        f_tip = self.N_b/2*(self.R-r)/(r*np.sin(inflow_angle))
        f_root = self.N_b/2*(r-self.R_c)/((self.R-r)*np.sin(inflow_angle))
        # f = f_tip*f_root
        F_tip = np.nan_to_num(2/np.pi*np.arccos(np.exp(-f_tip)), nan = 0)
        F_root = np.nan_to_num(2/np.pi*np.arccos(np.exp(-f_root)), nan = 0)
        F = F_tip*F_root
        
        U = (r*self.rpm)/np.cos(inflow_angle)
        alpha = self.twist(r) + self.collective - inflow_angle
        C_l = self.Cl_lookup(alpha)*F
        C_d = self.Cd_lookup(alpha)
        C_dF = C_d*F
        
        C_x = C_l*np.cos(inflow_angle) - C_dF*np.sin(inflow_angle)
        
        dT = self.N_b/2*rho*U**2*self.chord(r)*C_x
        return dT, alpha, C_l, C_d

    def eval_dP_BET(self, r, inflow_angle, U_climb):
        U = (r*self.rpm)/np.cos(inflow_angle)
        alpha = self.twist(r) + self.collective - inflow_angle
        C_t = self.Cl_lookup(alpha)*np.sin(inflow_angle) + self.Cd_lookup(alpha)*np.cos(inflow_angle)
        return self.N_b/2*rho*U**2*self.chord(r)*C_t*r*self.rpm
    
    def eval_dQ_BET(self, r, rpm, collective, inflow_angle):
        return self.eval_dT_BET(r, rpm, collective, inflow_angle)[0]*r
    
    def solve_T(self, rpm, collective, U_climb):
        return scipy.integrate.quad(lambda r: self.eval_dT_mom(r, rpm, self.solve_inflow_angle(r, rpm, collective, U_climb)), 0, self.R)

    def solve_P(self, rpm, collective, U_climb):
        # return scipy.integrate.quad(lambda r: self.eval_dP_BET(r, rpm, collective, self.solve_inflow_angle(r, rpm, collective, U_climb)), 0, self.R)
        return scipy.integrate.quad(lambda r: self.eval_dP_BET(r, rpm, collective, self.solve_inflow_angle(r, rpm, collective, U_climb)), 0, self.R)

    def solve_collective(self, rpm, T, U_climb):
        return scipy.optimize.fsolve(lambda collective: T - self.solve_T(rpm, collective, U_climb)[0], 0 )
