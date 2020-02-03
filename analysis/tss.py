import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from sympy.polys import poly
from sympy.abc import X,A,Y,B,T,D
import sys
from mpl_toolkits.mplot3d import Axes3D

import artem_test as art
from SMP.spin_motion_perturbation import SMPAnalysis
from analysis import load_davecs

VARS = ['X','A','Y','B','T','D']

def plot_3D_opt_cases(var, tilt_case):
    data_dir = '../data/TSS/{}/{}'.format(tilt_case, 'UNOPT')
    SMPAnalysis.fetch_from(data_dir+'/DUMMY_BOTH')
    case = SMPAnalysis('')
    z = case[VARS]
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    for i, opt_case in enumerate(['UNOPT', 'OPTIM']):
        data_dir = '../data/TSS/{}/{}'.format(tilt_case, opt_case)       
        st = load_davecs(data_dir)
        ax = fig.add_subplot(2,2,i+1, projection='3d')
        ax.set_xlabel('Y [mm]'); ax.set_ylabel('B'); ax.set_title(var + ' ({}, {})'.format(tilt_case, opt_case))
        var_vals = np.array([st[var](v) for v in z.T]).T
        for j, v in enumerate(var_vals.T):
            ax.plot(z['Y'][:,j], z['B'][:,j], var_vals[:,j])
        art.form(ax)
    for i, opt_case in enumerate(['UNOPT', 'OPTIM']):
        data_dir = '../data/TSS/{}/{}'.format(tilt_case, opt_case)       
        ax = fig.add_subplot(2,2,i+3, projection='3d')
        ax.set_xlabel('Y [mm]'); ax.set_ylabel('B')
        SMPAnalysis.fetch_from(data_dir+'/DUMMY_BOTH')
        case = SMPAnalysis('')
        z = case[VARS]
        var_vals = case.stunee[var]
        for j, v in enumerate(var_vals.T):
            ax.plot(z['Y'][:,j], z['B'][:,j], var_vals[:,j])
        art.form(ax)

def plot_3D(var, tilt_case, opt_case):
    data_dir = '../data/TSS/{}/{}'.format(tilt_case, opt_case)
    SMPAnalysis.fetch_from(data_dir+'/DUMMY_BOTH')
    case = SMPAnalysis('')
    z = case[VARS]
    st = load_davecs(data_dir)
    po_vals = case.stunee[var]
    py_vals = np.array([st[var](v) for v in z.T]).T
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel('Y [mm]'); ax.set_ylabel('B'); ax.set_title(var + ' ({}, {}, {})'.format(tilt_case, opt_case, 'POLVAL'))
    for i, v in enumerate(po_vals.T):
        ax.plot(z['Y'][:,i], z['B'][:,i], po_vals[:,i])
    art.form(ax)
    ax = fig.add_subplot(122, projection='3d')
    ax.set_xlabel('Y [mm]'); ax.set_ylabel('B'); ax.set_title(var + ' ({}, {}, {})'.format(tilt_case, opt_case, 'PYTHON'))
    for i, v in enumerate(py_vals.T):
        ax.plot(z['Y'][:,i], z['B'][:,i], py_vals[:,i]) 
    art.form(ax)

def plot_casewise(var):
    f, ax = plt.subplots(1,4, sharey=True)
    ax[0].set_ylabel(var + ' ({})'.format(TILTCASE))
    CASE = {}
    for cnv, v in enumerate(['NOT_B', 'NOT_Y', 'BOTH', 'NEITHER']):
        SMPAnalysis.fetch_from(DATA_DIR+'/DUMMY_'+v)
        CASE[v] = SMPAnalysis('')
        # for j in range(CASE[v].pray.shape[0]):
        z = CASE[v][:,1][VARS]
        #     if v=='NOT_B':
        #         z['B'] *= 0
        #     if v=='NOT_Y':
        #         z['Y'] *= 0
        #     if v=='NEITHER':
        #         z['Y'] *= 0
        #         z['B'] *= 0
        ax[cnv].plot(z['Y']*1e3, ST[var](z), '--.', label='PYTHON')
        ax[cnv].plot(CASE[v][:,1:]['Y'][1:]*1e3, CASE[v].stunee[var][1:,1:], '--.', label = 'POLVAL')
        ax[cnv].legend(); ax[cnv].set_title(v); ax[cnv].grid()
        ax[cnv].set_xlabel('Y [mm]')

def plot_main(z, x='Y', y='B', var='mu0', title=''):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    if title!='':
        ax.set_title('Trajectory: '+title)
    ax.set_xlabel(x+' [mm]'); ax.set_ylabel(y+' [mrad]')
    ax.plot(z[x]*1e3, z[y]*1e3)
    art.form(ax)
    ax = fig.add_subplot(122, projection='3d')
    ax.set_xlabel(x+' [mm]'); ax.set_ylabel(y+' [mrad]'); ax.set_title(var)
    for traj in z.T[1:]:
        ax.plot(traj[x]*1e3, traj[y]*1e3, ST[VAR](traj), label='{:4.2} [mm]'.format(traj[x][0]*1e3))
    ax.legend()
    art.form(ax)



if __name__ == '__main__':
    TILTCASE = sys.argv[1].upper() if len(sys.argv)>1 else 'IMPERFECT'
    OPTCASE = sys.argv[2].upper() if len(sys.argv)>2 else 'UNOPT'
    DATA_DIR = '../data/TSS/{}/{}'.format(TILTCASE, OPTCASE)
    VAR = sys.argv[3] if len(sys.argv)>3 else 'mu0'
    ST = load_davecs(DATA_DIR)
    
    SMPAnalysis.fetch_from(DATA_DIR+'/DUMMY_BOTH')
    case = SMPAnalysis('')
    z = case[VARS]

    var_X, var_Y = 'Y','B'

    z0 = np.zeros_like(z)
    ## Trajectory: line
    z0[var_X] = z[var_X]
    plot_main(z0, var_X,var_Y, VAR, 'line')
    
    ## Trajectory: (cos, sin)
    t = np.linspace(0,1,51)
    xmax = z[var_X].max()
    ymax = z[var_Y].max()
    nray = z.shape[1]
    z0[var_X] = np.outer(np.linspace(xmax/nray,xmax,nray), np.cos(2*np.pi*3*t)).T
    z0[var_Y] = np.outer(-1*np.linspace(ymax/nray, ymax, nray), np.sin(2*np.pi*3*t)).T
    plot_main(z0, var_X,var_Y, VAR, '(cos, sin)')

    ## Trajectory: real
    z0[var_X] = z[var_X]
    z0[var_Y] = z[var_Y]
    plot_main(z0, var_X,var_Y, VAR, 'data')
