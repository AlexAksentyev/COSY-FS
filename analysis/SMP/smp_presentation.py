# here i model the frequency variation introduced into polarimetry data as a result of
# the spin precession axis motion caused by betatron oscillations

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from scipy.optimize import curve_fit
import spin_motion_perturbation as smp
import analysis as ana
from importlib import reload
reload(smp); reload(ana)


OUT_DIR = '../../EDM/Reports/PhD/img/spin_axis_motion/long/'

def fit_reference(case):
    pguess = smp.guess_freq(case.P['t'], case.P['Y'])
    popt, pcov = curve_fit(lambda x, f: np.sin(2*np.pi*f* x), case.P['t'], case.P['Sy0'], p0 = pguess)
    perr = np.sqrt(np.diag(pcov))

    return ana.Estimate(popt[0], perr[0])

def plot_double_res(case):
    fits = case.Fits.fitted
    t = case.P['t']
    Sy0 = case.P['Sy0']
    Py = case.P['Y']
    plt.figure()
    plt.plot(t, Py-Sy0, '--b', label='vs CO')
    plt.plot(t, Py-fits, '--r', label='vs fits')
    plt.ylabel('Vertical polarization residual')
    plt.xlabel('time [sec]')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True)

def print_table(case_cw, case_ccw, offset):
    cw0est = fit_reference(case_cw)
    ccw0est = fit_reference(case_ccw)
    
    estis = np.array([case_cw.Freq.val, case_ccw.Freq.val, cw0est.val, ccw0est.val])
    ses = np.array([case_cw.Freq.se, case_ccw.Freq.se, cw0est.se, ccw0est.se])
    deltas = estis - offset
    
    print('Offset: {:8.5f}; deltas: {:4.7e}, {:4.7e}, {:4.9e}, {:4.9e}'.format(offset, *deltas))
    print('SE: {:4.0e}, {:4.0e}, {:4.0e}, {:4.0e}'.format(*ses))
    return cw0est, ccw0est


############################## UNIFORM BEAMS ##############################

## Everything to do with the residual plots
if True:
    smp.SMPAnalysis.fetch_from('../data/SMP/UNIFORM_LONG/')
    cw_uni = smp.SMPAnalysis('CW', full=False)
    
    cw_uni.fit_polarization()
    cw_uni.Fits.residual_vs_fit(fmt='.r')
    plt.savefig(OUT_DIR+'CW_LONG_residual_vs_fit.png')
    cw_uni.Fits.residual_vs_var('t', xlab='time [sec]', fmt='--.r')
    plt.savefig(OUT_DIR+'CW_LONG_residual_vs_time.png')
    plot_double_res(cw_uni)
    plt.savefig(OUT_DIR+'CW_LONG_double_res_full.png')
    
## Table data
if True:
    ccw_uni = smp.SMPAnalysis('CCW', full=False)
    ccw_uni.fit_polarization()
    cw0est_u, ccw0est_u = print_table(cw_uni, ccw_uni, 360.90365)

############################## SPIN TUNE AND N BAR #######################
i_x = np.arange(2, cw_uni.NRAY, 4); i_y = i_x + 1;
i_d = i_y + 1; i_xy = i_d + 1
pids = {'A': i_xy[15], 'B':i_xy[200], 'C':i_xy[799]}

smp.plot_three(cw_uni, pids)

############################## GAUSSIAN BEAMS ##############################
smp.SMPAnalysis.fetch_from('../data/SMP/GAUSS_LONG/')
cw_gau = smp.SMPAnalysis('CW', False)
ccw_gau = smp.SMPAnalysis('CCW', False)

## Initial beam histograms
smp.plot_beam_hists(cw_gau, ccw_gau, OUT_DIR)

## Table data
cw_gau.fit_polarization()
ccw_gau.fit_polarization()
cw0est_g, ccw0est_g = print_table(cw_gau, ccw_gau, 360.9036)

## Moving frame fit
if False:
    cw_gau.moving_fit(frame_size=250)
    ii = cw_gau.MF_Freq['se']<1e-2
    se = cw_gau.MF_Freq['se'][ii]
    est = cw_gau.MF_Freq['est'][ii]
    
    plt.errorbar(range(len(est)), est, yerr=se)
    plt.ylabel('Estimate [Hz]')
    plt.xlabel('Point #')
    plt.title('Run case: CW (Gauss_long)')
