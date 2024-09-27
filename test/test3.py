import numpy as np
import scipy
import matplotlib.pyplot as plt
from BEMT import Rotor

rho = 1
U_climb = 0

    
if __name__ == "__main__":    
    R = 2.5 # ft
    c = 2/12 # ft
    rpm = 960
    collectives = np.linspace(0,15,16)
    
    plt.figure()
    for N_b in [2]:
        rotor = Rotor(c, R, N_b)
        print(rotor.solidity)
        rotor.solve_collective(rpm, 0.015*(rho*rotor.A*(rpm*rotor.R)**2), U_climb)
