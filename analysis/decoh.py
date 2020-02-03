#!/usr/bin/env python3.6
#corresponds to fox script: scripts/BNL/decoh_optim.fox

import numpy as np
from functools import reduce
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os; import sys
from analysis import NLF, load_davecs
from CALIB.calib_danf import MU_TYPE
from sympy import Poly, symbols

VARS = ['X','A','Y','B','T','D']
X,A,Y,B,T,D = symbols(VARS)
SETUP = 'BNL'


plt.ion()
font = {'size'   : 14}
plt.rc('font', **font)

HOME_DIR = os.environ['HOME']+'/'
OUT_DIR = HOME_DIR+'REPOS/COSYINF/img/'+SETUP+'/decoh/'
os.system("mkdir -p "+OUT_DIR)

DATADIR =  HOME_DIR + 'REPOS/COSYINF/data/'+SETUP+'/decoh/'

TBL_TYPE = list(zip(['mrk','pid'], [object, int])) + MU_TYPE

def plot_stat(stat_name='mu0'):
    plt.figure()
    x = pray[var_bunch][ind[var_bunch]] * 1e3 # turn meters to mm
    plt.plot(x, unopt[stat_name][ind[var_bunch]], label='unopt')
    plt.plot(x, optim[var_case][stat_name][ind[var_bunch]], label='optim')
    plt.ticklabel_format(style='sci', axis='both', scilitmits=(0,0))
    plt.legend(); plt.grid()
    plt.title('case {}, bunch {}'.format(var_case, var_bunch))
    plt.xlabel(var_bunch+' [mm]'); plt.ylabel(stat_name)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True)


if __name__ == '__main__':

    pray = NLF['PRAY'][1](DATADIR+'PRAY.dat')
    pspi = NLF['PRAY'][1](DATADIR+'PSPI.dat')


    nray = pray.shape[0]
    ind = dict(X = np.arange(2, nray, 3),
               Y = np.arange(3, nray, 3),
               D = np.arange(4, nray, 3))
    
    unopt = np.loadtxt(DATADIR+'STUNEE:UNOPT.dat', dtype = TBL_TYPE)
    unopt_da_s = load_davecs(DATADIR, 'UNOPT')
    optim = {};
    opt_da_s = {}
    for var in ['X','Y','D', 'XY', 'YD', 'XYD']:
        optim[var] = np.loadtxt(DATADIR+'STUNEE:OPTIM_{}.dat'.format(var), dtype = TBL_TYPE)
        opt_da_s[var] = load_davecs(DATADIR, 'OPTIM_'+var)

    var_case=sys.argv[1]
    var_bunch=sys.argv[2] if len(sys.argv)>2 else var_case

    unF = unopt_da_s['mu0']
    opF = opt_da_s[var_bunch]['mu0']
    oppY = opF.poly.eval(dict(X=0,A=0,B=0,T=0,D=0))

    plt.figure()
    x = pray[var_bunch][ind[var_bunch]]
    plt.plot(x*1e3, unopt['mu0'][ind[var_bunch]], label='unopt')
    plt.plot(x*1e3, optim[var_case]['mu0'][ind[var_bunch]], label='optim')
    plt.ticklabel_format(style='sci', axis='both', scilitmits=(0,0))
    plt.legend(); plt.grid()
    plt.title('POLVAL EVAL: case {}, bunch {}'.format(var_case, var_bunch))
    plt.xlabel(var_bunch+' [mm]'); plt.ylabel('spin tune')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True)

    z = np.zeros(x.shape[0], dtype = list(zip(VARS, [float]*6)))
    z[var_bunch] = x
    plt.figure()
    plt.plot(x*1e3, unF(z), label='unopt')
    plt.plot(x*1e3, opF(z), label='optim')
    plt.legend(); plt.grid()
    plt.title('PYTHON EVAL: case {}, bunch {}'.format(var_case, var_bunch))
    plt.xlabel(var_bunch+' [mm]'); plt.ylabel('spin tune')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True)

