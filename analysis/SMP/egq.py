import analysis as ana
import spin_motion_perturbation as smp
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from importlib import reload
from functools import reduce
reload(smp); reload(ana)

#################### DEFINITIONS ####################
Analysis = ana.Analysis
Estimate = ana.Estimate

land = lambda *P: reduce(np.logical_and, P)
lor = lambda *P: reduce(np.logical_or, P)

def delta(dat, var, bound=None):
    x = dat.pray['X']
    y = dat.pray['Y']
    if bound is None:
        bound = np.max([abs(x),abs(y)])
    i_xp = land(x>0, abs(x)<bound)
    i_xn = land(x<0, abs(x)<bound)
    i_x = i_xn+i_xp
    i_yp = land(y>0, abs(y)<bound)
    i_yn = land(y<0, abs(y)<bound)
    i_y = i_yn+i_yp

    variable = dat.stunee[var]

    d_in_x = variable[i_xp] - np.flip(variable[i_xn], axis=0)
    d_in_y = variable[i_yp] - np.flip(variable[i_yn], axis=0)

    nmin = np.min([variable[i_x].shape[0], variable[i_y].shape[0]])
    d_xy = variable[i_x][:nmin] - variable[i_y][:nmin]

    stats = lambda d: (d.mean(), d.std()/np.sqrt(d.shape[0]))

    return Estimate(*stats(d_in_x)), Estimate(*stats(d_in_y)), Estimate(*stats(d_xy))

#################### PEPARATIONS ####################
Analysis.fetch_from('../data/EGQ/')
dat = Analysis.load('QUE', ana.NLF['STUNEE'], ana.NLF['PRAY'])#, ana.NLF['TRPSPI'], ana.NLF['TRPRAY'])

#################### ANALYSIS ####################
def analyze(bound=None, show=0):
    d = np.empty(4, dtype = list(zip(['x','y','xy'], [Estimate]*3)))
    for i, var in enumerate(dat.stunee.dtype.names[1:]):
        d[i] = delta(dat, var, bound)

    print('GROUP X: ',  d['x'][show], '')
    print('GROUP Y: ',  d['y'][show], '')
    print('GROUP XY: ', d['xy'][show])
    return d

mu0 = dat.stunee['mu0']
nx = dat.stunee['nx']
ny = dat.stunee['ny']
nz = dat.stunee['nz']
x = dat.pray['X']
y = dat.pray['Y']
i_x = x!=0
i_y = y!=0
x = x[i_x]
y = y[i_y]

gx = mu0[i_x]
gy = mu0[i_y]

def triple_plot(var, diff=False):
    x = dat.pray['X']
    y = dat.pray['Y']
    i_x = x!=0
    i_y = y!=0
    x = x[i_x]
    y = y[i_y]
    gx = dat.stunee[var][i_x]
    gy = dat.stunee[var][i_y]
    f, ax = plt.subplots(3,1)
    ax[0].set_title(var)
    if not diff:
        ax[0].plot(x, gx, '-b', label='GX') 
        ax[0].plot(y, gy, '-r', label='GY')
    else:
        ax[0].plot(gx-gy, '-k', label='GX-GY')
    ax[0].set_xlabel('X/Y')
    ax[0].legend()
    ax[0].axvline(x=0, color='gray', linestyle='--')
    ax[0].ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    ax[1].plot(x[x>0],gx[x>0] - np.flip(gx[x<0], axis=0), '-b', label='GX')
    ax[1].legend()
    ax[1].set_xlabel('X')
    ax[1].ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    ax[2].plot(y[y>0],gy[y>0] - np.flip(gy[y<0], axis=0), '-r', label='GY')
    ax[2].legend()
    ax[2].set_xlabel('Y')
    ax[2].ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    dgxdx = np.diff(gx)/np.diff(x)
    xzci = np.where(np.diff(np.sign(dgxdx)))[0]
    dgydy = np.diff(gy)/np.diff(y)
    yzci = np.where(np.diff(np.sign(dgydy)))[0]
    ax[0].axvline(x=x[xzci[0]], color='b')
    ax[0].axvline(x=x[yzci[0]], color='r')


