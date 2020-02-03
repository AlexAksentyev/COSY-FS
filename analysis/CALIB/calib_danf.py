import numpy as np
import matplotlib.pyplot as plt
import os, sys
from analysis import Analysis, NLF

HOMEDIR = os.environ['HOME']

MU_TYPE = list(zip(['mu0','nx', 'ny', 'nz'], [float]*4))

TOF = 1e-6
plt.ion()

wx = lambda tbl: 2*np.pi*tbl['mu0']*tbl['nx']/TOF
wy = lambda tbl: 2*np.pi*tbl['mu0']*tbl['ny']/TOF
wz = lambda tbl: 2*np.pi*tbl['mu0']*tbl['nz']/TOF

def main_plot(case, edm, pids):
    plt.plot(dwy[case,edm,pids], dwx[case,edm,pids], '.')
    plt.xlabel('dWy'); plt.ylabel('dWx')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid()

def ninj(tbl):
    return [len(np.unique(e)) for e in (tbl['i'], tbl['j'])]

def load_gauss(data_dir=None):
    data_dir = DATADIR if data_dir is None else data_dir
    Analysis.fetch_from(data_dir)
    gauss_fnames = Analysis.list_files('GAUSS')
    gauss = np.empty(len(gauss_fnames), dtype = [('i', int), ('j', int), ('mean', float), ('std', float)])

    for ind, fname in enumerate(gauss_fnames):
        s = fname.replace("X",":").replace("S",":").replace(".",":").split(":")
        i = int(s[-3]); j = int(s[-2])
        tilts = np.loadtxt(DATADIR+fname)
        gauss[ind] = i, j, tilts.mean(), tilts.std()
    return np.sort(gauss, axis=0, order='i')


if __name__ == '__main__':
    SETUP = sys.argv[1]
    DATADIR = HOMEDIR + '/REPOS/COSYINF/data/calib/{}/'.format(SETUP)
    pray = NLF['PRAY'][1](DATADIR+'PRAY.dat')
    pspi = NLF['PRAY'][1](DATADIR+'PSPI.dat')
    gauss = load_gauss()
    gauss.shape = ninj(gauss)

    nray = pray.shape[0]
    i_x = np.arange(1, nray, 5)
    i_y = i_x + 1
    i_d = i_y + 1
    i_sx = i_d + 1
    i_sy = i_sx + 1

    cw = np.loadtxt(DATADIR+'STUNE:CW.dat', dtype = list(zip(['i','j'], [int]*2)) + MU_TYPE)
    cw.shape = ninj(cw)
    ccw = np.loadtxt(DATADIR+'STUNE:CCW.dat', dtype = list(zip(['i','j'], [int]*2)) + MU_TYPE)
    ccw.shape = ninj(ccw)
    cwe = np.loadtxt(DATADIR+'STUNEE:CW.dat', dtype = list(zip(['i','j', 'pid'], [int]*3)) + MU_TYPE)
    cwe.shape = ninj(cwe)+[-1]
    ccwe = np.loadtxt(DATADIR+'STUNEE:CCW.dat', dtype = list(zip(['i','j', 'pid'], [int]*3)) + MU_TYPE)
    ccwe.shape = ninj(ccwe)+[-1]
    gauss.sort(order='i')

    
    wxcw = wx(cwe); wycw = wy(cwe); wzcw = wz(cwe)
    wxccw = wx(ccwe); wyccw = wy(ccwe); wzccw = wz(ccwe)

    dwx = wxcw - wxccw; dwy = wycw - wyccw; dyz = wzcw - wzccw
   
    main_plot(0, 0, i_x)
    main_plot(0, 0, i_y)
    main_plot(0, 0, i_d)
