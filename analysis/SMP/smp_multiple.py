# here i model the frequency variation introduced into polarimetry data as a result of
# the spin precession axis motion caused by betatron oscillations

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import sys
from scipy.optimize import curve_fit
from scipy.stats import norm
from spin_motion_perturbation import SMPAnalysis

OUT_DIR = '../../EDM/Reports/PhD/img/spin_axis_motion/multiple/'
STAT_TYPE = [('est', float), ('se', float)]

class RunContainer:
    def __init__(self, *marker_list):
        self.run_num = len(marker_list)
        self.shape = (self.run_num,)
        self.run_list = np.empty(self.run_num, dtype=object)
        for i, mrk in enumerate(marker_list):
            self.run_list[i] = SMPAnalysis(mrk)

    def fit_polarization(self):
        self.Freq = np.empty(self.run_num, dtype = STAT_TYPE)
        for i, run in enumerate(self.run_list):
            run.fit_polarization()
            self.Freq[i] = run.Freq.val, run.Freq.se

    def __getitem__(self, index):
        return self.run_list[index]

    def runs(self, from_=0, to_=None):
        if to_ is None or to_ > self.run_num:
            to_= self.run_num
        for run_id in range(from_, to_):
            yield self[run_id]

def error_stat(one_tbl, two_tbl):
    """CW-CCW frequency estimate difference and standard error"""
    err = one_tbl['est'] - two_tbl['est']
    se = np.sqrt(one_tbl['se']**2 + two_tbl['se']**2)
    return np.array(list(zip(err, se)), dtype=STAT_TYPE)

def centroid_diff(one, two):
    DZ = np.empty(one.shape, dtype = list(zip(['X','Y','D'], [float]*3)))
    for i, run in enumerate(one):
        dz = run.centroid - two[i].centroid # change thiese 2 lines
                                         # for centroid.as_tuple
        DZ[i] = dz.X, dz.Y, dz.D
    return DZ

def plot_errstat_estimates(reg_var='Y'):
    lin = lambda x, a,b: a+b*x
    ii = errstat['se']<1e-2
    est = errstat['est'][ii]
    se = errstat['se'][ii]
    scale = 1e3 if reg_var != 'D' else 1
    cdii = centdiff[ii]
    x = cdii[reg_var]*scale
    popt, pcov = curve_fit(lin, x, est)
    perr = np.sqrt(np.diag(pcov))
    pval = norm.cdf(-abs(popt[1]/perr[1]))
    plt.figure()
    plt.errorbar(x, est, yerr=se, fmt='.', label='estimates')
    plt.plot(x, lin(x, *popt), '--r', label='fit')
    xlab = 'Delta mean ' + reg_var + ' [mm]' if reg_var != 'D' else reg_var
    plt.xlabel(xlab); plt.ylabel('CW - CCW frequency estimate difference [Hz]')
    plt.title('Slope = {:4.2e}+-{:4.2e}; P(Slp=0) = {:4.2f}%'.format(popt[1], perr[1], pval*100))
    plt.legend()
    plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    plt.savefig(OUT_DIR+'freq_estimates_vs_centroid_diff_{}.png'.format(reg_var))
    
    

if __name__ == '__main__':

    INDIR = '../data/SMP/{}/'.format(sys.argv[1]) if len(sys.argv)>1 else '../data/SMP/'
    SMPAnalysis.fetch_from(INDIR)
    cw_mrk = sorted(SMPAnalysis.list_markers('CW'), key=lambda x: float(x[2:]))
    ccw_mrk = sorted(SMPAnalysis.list_markers('CCW'), key=lambda x: float(x[3:]))

    cw_arr = RunContainer(*cw_mrk)
    ccw_arr = RunContainer(*ccw_mrk)
    del cw_mrk, ccw_mrk

    cw_arr[2].plot_polarization()
    plt.savefig(OUT_DIR+'CW3_polarization.png')

    cw_arr.fit_polarization()
    ccw_arr.fit_polarization()

    errstat = error_stat(cw_arr.Freq, ccw_arr.Freq)
    centdiff = centroid_diff(cw_arr, ccw_arr)
    
    plot_errstat_estimates('X')
    plot_errstat_estimates('Y')
    plot_errstat_estimates('D')
