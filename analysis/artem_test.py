
# Artem wants to see the dependence of the Frequency estimate on the centroid offset from the CO
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import sys
from scipy.optimize import curve_fit
from scipy.stats import norm
from SMP.spin_motion_perturbation import SMPAnalysis, guess_freq, guess_phase
import statsmodels.api as sm
from decoh import MU_TYPE
from analysis import Bundle, NLF
import inspect

form = lambda ax: ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)

OUT_DIR = '../img/Artem/'
STAT_TYPE = [('est', float), ('se', float)]

lin = lambda x, a,b: a+b*x
quad = lambda x, a,b,c:  a + b*x + c*x**2
sine = lambda x, a,f,p: a*np.sin(2*np.pi*f*x + p)
jitsine = lambda x, a0,a1,f0,f1,p: (a0-a1*abs(np.sin(2*np.pi*f1*x)))*np.sin(2*np.pi*f0*x+p)

def fit_func(ff, x, y, p_guess):
    print(p_guess)
    popt, pcov = curve_fit(ff, x, y, p0=p_guess)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def fit_sine(x,y):
    p_guess = [y.max(), guess_freq(x,y), guess_phase(x,y)]
    print(p_guess)
    return fit_func(sine, x, y, p_guess)

def res_fit(x,y):
    popt, perr = fit_sine(x,y)
    model = sine(x, *popt)
    return y-model, model

class RunContainer:
    def __init__(self, *marker_list):
        self.count = len(marker_list)
        self.shape = (self.count,)
        self._mrk_list = marker_list
        self.run_list = np.empty(self.count, dtype=object)
        self.fit_parameters=Bundle()
        for i, mrk in enumerate(marker_list):
            self.run_list[i] = SMPAnalysis(mrk)

    def fit_polarization(self, var='Y', fit_fun=sine, ini_guess=None):
        guess_keys = ['Ampl', 'Freq', 'Phase'] if ini_guess is None else list(ini_guess.keys())

        par_row = dict()
        for key in guess_keys:
            setattr(self.fit_parameters, key, np.empty(self.count, dtype = STAT_TYPE))
            par_row[key] = getattr(self.fit_parameters, key)

        for i, run in enumerate(self.run_list):
            run.fit_polarization(var, fit_fun=fit_fun, ini_guess=ini_guess)
            for name in run.model.parameters:
                par = getattr(run.model.parameters, name)
                par_row[name][i] = par.val, par.se                

    def stunee_stat(self):
        stats = np.empty((self.count, 4), dtype = STAT_TYPE)
        for i, run in enumerate(self.run_list):
            mu0mean = run.stunee['mu0'].mean()
            mu0std = run.stunee['mu0'].std()
            stats[i, 0] = mu0mean, mu0std
            for j, var in enumerate(['nx', 'ny', 'nz'], 1):
                nmean = run.stunee[var].mean()
                nstd = run.stunee[var].std()
                stats[i, j] = nmean, nstd
        return stats

    def __getitem__(self, index):
        return self.run_list[index]

    def __repr__(self):
        return str(self.shape)

    def runs(self, from_=0, to_=None):
        if to_ is None or to_ > self.count:
            to_= self.count
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
        dz = run.centroid - two[i].centroid 
        DZ[i] = dz.X, dz.Y, dz.D
    return DZ

def centroid_off(one):
    DZ = np.empty(one.shape, dtype = list(zip(['X','Y','D'], [float]*3)))
    for i, run in enumerate(one):
        dz = run.centroid
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

def plot_freq_estimates(arr, reg_var='Y', ff=lin):
    ii = arr.fit_parameters.Freq['se']<5e-6
    est = arr.fit_parameters.Freq['est'][ii]
    se = arr.fit_parameters.Freq['se'][ii]
    scale = 1e3 if reg_var != 'D' else 1
    cdii = coff[ii]
    x = cdii[reg_var]*scale
    popt, pcov = curve_fit(ff, x, est, sigma=None, absolute_sigma=False)
    perr = np.sqrt(np.diag(pcov))
    pval = norm.cdf(-abs(popt[1]/perr[1]))
    plt.figure()
    plt.errorbar(x, est, yerr=se, fmt='.', label='estimates')
    plt.plot(x, ff(x, *popt), '--r', label='fit')
    xlab = 'mean ' + reg_var + ' offset [mm]' if reg_var != 'D' else reg_var
    plt.xlabel(xlab); plt.ylabel('Frequency estimate [Hz]')
    plt.title('Slope = {:4.2e}+-{:4.2e}; P(Slp=0) = {:4.2f}%'.format(popt[1], perr[1], pval*100))
    plt.legend()
    plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)

def normalize(stat):
    return (stat-stat.mean(axis=0))/stat.std(axis=0)

def plot_all_stunees(xvar, xlab, st_stat):
    mu0, nx, ny, nz = normalize(st_stat['est']).T
    mu0std, nxstd, nystd, nzstd = st_stat['est'].std(axis=0)
    plt.figure()
    plt.plot(xvar, nx, '--.', label='nx ({:4.2e})'.format(nxstd))
    plt.plot(xvar, ny, '--.', label='ny ({:4.2e})'.format(nystd))
    plt.plot(xvar, nz, '--.', label='nz ({:4.2e})'.format(nzstd))
    plt.plot(xvar, -mu0, '--.', label='-mu0 ({:4.2e})'.format(mu0std))
    plt.legend()
    plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    plt.ylabel('RMS-normalized mean statistic')
    plt.xlabel(xlab)
    
def load_case(dir_name):
    dir_path = '../data/ARTEM_TEST/'+dir_name
    SMPAnalysis.fetch_from(dir_path)
    cw_mrk = sorted(SMPAnalysis.list_markers('CW'), key=lambda x: float(x[2:]))
    cw_arr = RunContainer(*cw_mrk)
    if len(SMPAnalysis.list_markers('DECOH'))>0:
        decoh_data = np.loadtxt(dir_path+'/STUNEE:DECOH.dat', dtype=[('pid',int)]+MU_TYPE)
        decoh_pray = NLF['PRAY'][1](dir_path+'/PRAY:DECOH.dat')
        cw_arr.decoh = Bundle(data=decoh_data, pray=decoh_pray)
    return cw_arr

if __name__ == '__main__':
    CASE = sys.argv[1]

    cw_arr = load_case(CASE)

    cw_arr.fit_polarization('Y')
    coff = centroid_off(cw_arr)
    st_stat = cw_arr.stunee_stat()

    X = sm.add_constant(st_stat['est'])
    model = sm.OLS(cw_arr.fit_parameters.Freq['est'], X).fit()
    print(model.summary())

    mu0, nx, ny, nz = normalize(st_stat['est']).T
    mu0std, nxstd, nystd, nzstd = st_stat['se'].T
    f = normalize(cw_arr.fit_parameters.Freq['est'])
    yoff = coff['Y']

    CASE = CASE.replace('/', '_')
    plot_freq_estimates(cw_arr, 'Y')
    plt.savefig('../img/Artem/freq_est_vs_y_offset_{}.png'.format(CASE))
    plot_all_stunees(yoff*1e3, 'Y offset [mm]', st_stat); plt.title(CASE)
    plt.savefig('../img/Artem/stune_stats_vs_y_offset_{}.png'.format(CASE))
    plot_all_stunees(cw_arr.Freq['est'], 'Frequency [Hz]', st_stat); plt.title(CASE)
    plt.savefig('../img/Artem/stune_stats_vs_freq_{}.png'.format(CASE))
    plt.figure()
    plt.plot(mu0, nx, '.', label='nx')
    plt.plot(mu0, ny, '.', label='ny')
    plt.legend()
    plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    plt.xlabel('normalized spin tune'); plt.ylabel('normalized n-bar'); plt.title(CASE)
    plt.savefig('../img/Artem/n-bar_on_spin_tune_{}.png'.format(CASE))
    
