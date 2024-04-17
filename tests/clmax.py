from pyXFOIL import XFOILCase
from pathlib import Path

case = XFOILCase(af_path = "/home/canativi/Documents/Low_Re_Validation/Koning_E387/e387.dat",
               Re = 5e5, Ma = 0, alpha = [2, 4, 6],
               iter = 50, panels = 301, 
               clean = True, run_path=f"{Path(__file__).parent}/run/", xfoil_path = "/home/canativi/Software/wfoil/bin/xfoil",
               verbose = True)
case.run()
alpha, CL_max = case.CL_max()
print(alpha, CL_max)