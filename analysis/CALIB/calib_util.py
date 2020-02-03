import numpy as np
import matplotlib.pyplot as plt
from analysis import STAT_TYPE, fit


def load_spnr(filename):
    spnr = np.loadtxt(Analysis.data_dir + filename)
    spnr.shape = (3,3,-1)
    return spnr

# **************************************************
def _w_fit(case, out_file=None):
    """Computes the spin component oscillation frequencies by fitting tracker data"""
    close_file=False
    if out_file is None:
        out_file = PdfPages(os.environ['HOME']+'/REPOS/COSYINF/report/spin_fits.pdf')
        close_file = True
    nray = case.trpspi.shape[1]
    para = np.empty((nray, 3), dtype=STAT_TYPE)
    for i, pcl in enumerate(case.trpspi.transpose()):
        fig, ax = plt.subplots(3,1, sharex=True)
        sx0 = pcl['S_X'][0]
        sy0 = pcl['S_Y'][0]
        sz0 = np.sqrt(1-sx0**2-sy0**2)
        sya = np.sqrt(1-sx0**2)
        # fitting Sy
        py, ey, _ = fit(pcl, lambda t, w, phi: sya*np.sin(w*t + phi), [1, np.pi], 'S_Y', ax[1])
        # print('Sy fit pars:', py)
        para[i,1] = py[0], ey[0]
        # fitting Sz; initial guess: Sy omega
        fitfun = lambda t, w, a: a*np.cos(w*t)
        if abs(sy0)>abs(sx0): # initial offset is in Sy, not Sx
            fitfun = lambda t, w, a: a - w*t
        pz, ez, _ = fit(pcl, fitfun, [py[0], sz0], 'S_Z', ax[2])
        para[i,2] = pz[0], ez[0]
        # fitting Sx
        px, ex, _ = fit(pcl, lambda t, w, phi: np.sin(w*t + phi), [1, pz[1]], 'S_X', ax[0])
        para[i,0] = px[0], ex[0]
        out_file.savefig(fig)
        plt.close()
    if close_file:
        out_file.close()
    return para

def omega_fit(lst):
    """Produces spin component oscillation frequencies [rad/sec]"""
    n_lst = len(lst); nray = lst[0].pray.shape[0]
    w = np.empty((n_lst, nray, 3), dtype=STAT_TYPE)
    for i, a in enumerate(lst):
        w[i] = _w_fit(a)           
    return w

# **************************************************
def dermean(case):
    from analysis import _der_mean, DER_ORDER
    nrec, nray = case.trpspi.shape
    nrec = int((nrec - nrec%DER_ORDER)/DER_ORDER)
    h = case.trpspi['iteration'][1, 0]*TOF
    der = np.empty((nray, 3, nrec), dtype = [('der', float), ('mean', float)])
    for i, pcl in enumerate(case.trpspi.T):
        for j, var in enumerate(['S_X', 'S_Y', 'S_Z']):
            d, m = _der_mean(pcl[var], h)
            der['der'][i,j,:] = d
            der['mean'][i,j,:] = m
    return der

def plot_mean_der(case, pids=[0,1,2]):
    dm = dermean(case)
    f, ax = plt.subplots(2,3)
    ttl = ['S_X', 'S_Y', 'S_Z']
    for i in range(3):
        ax[0,i].set_title(ttl[i])
        ax[0,i].plot(dm.T['mean'][:,i,pids], '--.')
        ax[1,i].plot(dm.T['der'][:,i,pids], '--.')
    ax[0,0].set_ylabel('mean')
    ax[1,0].set_ylabel('derivative')
    return f
