import numpy as np
import os, shutil, pathlib, subprocess, dask
import matplotlib.pyplot as plt
# plt.style.use("https://raw.githubusercontent.com/cnativid/MPL-Styles/main/default.mplstyle")

class PyXFOIL:
    "Class for an XFOIL run"
    def __init__(self, x = None, z = None, af_path = None, Re = 1e5, Ma = 0.03, alpha = [], CL = [], 
                 iter = 50, panels = None, # solver options
                 clean = False, run_path = "./run/pyxfoil", xfoil_path = "./bin/xfoil",
                 name = None, verbose = False, timeout = None):

        # check how the airfoil is being generated
        if af_path: # defined by airfoil data file
            self.name = open(af_path,"r").readline().strip()
            x, z = np.loadtxt(af_path, skiprows=1, unpack=True)
            
        if name:
            self.name = name
        else:
            self.name = "pyXFOIL Airfoil"
        
        self.x = x
        self.z = z
        le_index = x.argmin() # split top and bottom surfaces
        self.xtop = np.flip(x[:le_index+1])
        self.ztop = np.flip(z[:le_index+1])
        self.xbot = x[le_index:]
        self.zbot = z[le_index:]
        
        if run_path:
            self.run_path = run_path
        else:
            self.run_path = open(af_path).readline()
            
        self.Re = Re
        self.Ma = Ma
        self.alpha = alpha
        self.CL = CL
        
        self.iter = iter
        self.panels = panels
        
        self.clean = clean
        self.run_path = run_path
        case_path = f"{run_path}/Re{Re:1.3e}_Ma{Ma:.3f}"
        self.case_path = case_path
        self.polar_path = f"{case_path}/polar"
        self.cpdir_path = f"{case_path}/cp"
        self.bldir_path = f"{case_path}/bl"
        self.log_path = f"{case_path}/log.log"
        self.xfoil_path = str(pathlib.Path(xfoil_path).absolute())
        
        self.polar_loaded = False
        self.verbose = verbose
        self.timeout = timeout

    def run(self):
        x = self.x
        z = self.z
        Re = self.Re
        Ma = self.Ma
        run_path = self.run_path
        case_path = self.case_path
        panels = self.panels
        
        if not os.path.exists(run_path): # create the run path if it doesn't exist already
            os.system(f"mkdir {run_path}")
        else: # see if we want to remove all the cases
            if self.clean == True:
                os.system(f"rm -rf {run_path}")
                os.system(f"mkdir {run_path}")
            else:
                raise Exception("Solution files already exist. Set clean == True to overwrite.")
            
        # copy airfoil file 
        with open(f"{run_path}/af.dat", "w") as file:
            file.write(self.name)
            file.writelines([f"\n{x} {z}" for (x, z) in zip(x, z)])
            file.write('\n')
            
        # create input files
        for f in ["cp", "bl"]:
            try:
                os.makedirs(f"{case_path}/{f}")
            except:
                pass

        input = """plop
g f

load ../af.dat
"""
        if panels:
            input += f"ppar\nn {panels:g}\n\n\n"
        input += f"""oper 
v {Re}
m {Ma}
iter {self.iter:g}
pacc
polar
.bl
"""
        for alpha in self.alpha:
            input += (f"a {alpha:2.3f}\ncpwr ./cp/a{alpha:2.3f}\ndump ./bl/a{alpha:2.3f}\n")
        
        for CL in self.CL:
            input += (f"CL {CL:2.3f}\ncpwr ./cp/CL{CL:2.3f}\ndump ./bl/CL{CL:2.3f}\n")
            
        input += "\n\n\n\nquit"
        
        if self.verbose:
            stdout = open(f"{case_path}/log.log","w")
        else:
            stdout = subprocess.DEVNULL
        
        sp = subprocess.Popen(self.xfoil_path, cwd = f"{case_path}",
                                stdin=subprocess.PIPE,
                                stdout=stdout,
                                stderr=subprocess.DEVNULL,
                                )
        sp.communicate(input.encode("utf-8"), timeout = self.timeout)
    
    def CL_max(self):
        x = self.x
        z = self.z
        Re = self.Re
        Ma = self.Ma
        run_path = self.run_path
        case_path = self.case_path
        panels = self.panels
        
        if not os.path.exists(run_path): # create the run path if it doesn't exist already
            os.system(f"mkdir {run_path}")
        else: # see if we want to remove all the cases
            if self.clean == True:
                os.system(f"rm -rf {run_path}")
                os.system(f"mkdir {run_path}")
            else:
                raise Exception("Solution files already exist. Set clean == True to overwrite.")
            
        # copy airfoil file 
        with open(f"{run_path}/af.dat", "w") as file:
            file.write(self.name)
            file.writelines([f"\n{x} {z}" for (x, z) in zip(x, z)])
            file.write('\n')
            
        # create input files
        for f in ["cp", "bl"]:
            try:
                os.makedirs(f"{case_path}/{f}")
            except:
                pass

        input = """plop
g f

load ../af.dat
"""
        if panels:
            input += f"ppar\nn {panels:g}\n\n\n"
        input += f"""oper 
v {Re}
m {Ma}
iter {self.iter:g}
pacc
polar
.bl
aseq 0 20 1
\n\n\n\nquit"
"""
        
        if self.verbose:
            stdout = open(f"{case_path}/log.log","w")
        else:
            stdout = subprocess.DEVNULL
        
        sp = subprocess.Popen(self.xfoil_path, cwd = f"{case_path}",
                                stdin=subprocess.PIPE,
                                stdout=stdout,
                                stderr=subprocess.DEVNULL,
                                )
        sp.communicate(input.encode("utf-8"), timeout = self.timeout)
        
        polar = self.get_polar()
        idx = np.argmax(polar["CL"])
        return  polar["CL"][idx], polar["alpha"][idx]
    
    
    def get_polar(self):
        # alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr, Top_Itr, Bot_Itr
        # try:
        #     return self.polar
        # except:
        with open(self.polar_path, "r") as file: # check to see if any results converged
            [next(file) for _ in range(12)]
            if len(next(file, "")) == 0:
                return {}
            else:
                self.polar_loaded = True        
                polar_data = np.loadtxt(self.polar_path, skiprows = 12, unpack = True)
                self.polar = dict(zip(("alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr", "Top_Itr", "Bot_Itr"),
                                    np.loadtxt(self.polar_path, skiprows = 12, unpack = True)))
                return self.polar
    
    def get_vsx(self, alpha = None, CL = None):
        if alpha:
            fname = f"a{alpha:.3f}"
        elif CL:
            fname = f"CL{CL:.3f}"
        #      x          Cp  
        #    s        x        y     Ue/Vinf    Dstar     Theta      Cf       H       H*        P         m          K          tau         Di
        cp_data = np.loadtxt(f"{self.cpdir_path}/{fname}", skiprows = 1, unpack = True)
        bl_data = np.loadtxt(open(f"{self.bldir_path}/{fname}","r").readlines()[1:(len(cp_data[0])+1)], unpack = True, usecols=range(3, 7))
        vx = dict(
            zip(("x", "Cp", "Ue/Vinf", "Dstar", "Theta", "Cf", "H"), # , "H*", "P", "m", "K"
                            np.concatenate([cp_data,bl_data]))
            )
        return vx
        
    # def plot_Cp(self, cases = None, labels = None, linestyles = None):
    # def plot_Cp(self, cases = None):
        
    #     fig, (ax1, ax2) = plt.subplots(2,1, figsize = [5,5], gridspec_kw={'height_ratios': [7,1]})
    #     ax2.axis("equal")
    #     ax2.plot(self.x,self.z, 'k')
    #     ax2.fill_between(self.xtop, self.ztop, self.zbot, alpha = 0.3)
    #     ax1.invert_yaxis()
        
    #     if not cases:
    #         for cp_path in [f"{self.cpdir_path}/{case}" for case in os.listdir(self.cpdir_path)]:
    #             x, Cp = np.loadtxt(cp_path, unpack = True, skiprows = 1)
    #             ax1.plot(x, Cp)
    #     else:
    #         for cp_path in [f"{self.cpdir_path}/{case}" for case in cases]:
    #             x, Cp = np.loadtxt(cp_path, unpack = True, skiprows = 1)
    #             ax1.plot(x, Cp)

    #     ax1.set_xlabel("x/c")
    #     ax1.set_ylabel("C_p")
    #     ax1.spines['bottom'].set_position('zero')
    #     ax1.spines[['right', 'top']].set_visible(False)
    #     ax2.set_axis_off()
    #     plt.subplots_adjust(hspace=0)
    #     ax1.title.set_text("C_p vs x/c")
    #     plt.tight_layout()

    #     return fig, [ax1, ax2]
    
#     def bend(self):
#         input = f"""plop
# g f

# load af.dat
# bend

# quit
# """
#         sp = subprocess.Popen(self.xfoil_path, cwd = f"{self.run_path}",
#                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
#         out,_ = sp.communicate(input.encode("utf-8"))
#         outlist = out.decode("utf-8").splitlines()
#         return outlist[-7]


class XFOILCase(PyXFOIL):
    pass

if __name__ == "__main__":
    # get XFOIL result
    case1 = PyXFOIL(af_path = "/home/canativi/Documents/optTURNS/safe/Dec/gen0_small/clarky.dat",
                    Re = 1e5, Ma = 0.03, alpha = np.linspace(0,10,21), CL = [],
                 clean = True, run_path = "./run/pyxfoil", xfoil_path = "./bin/xfoil")
    case1.run()
