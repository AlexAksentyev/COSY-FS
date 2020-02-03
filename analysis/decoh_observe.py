from SMP import spin_motion_perturbation as smp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from analysis import Bundle, Estimate

plt.ion()
font = {'size'   : 16}
plt.rc('font', **font)

form = lambda ax:  ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
Analysis = smp.SMPAnalysis
Model = smp.Model
guess_freq = smp.guess_freq
guess_phase = smp.guess_phase

def fit_reference(case, var='Y'):
    ff = lambda x, a,f,p: a*np.sin(2*np.pi*f*x + p)
    sp = case.P['S{}0'.format(var.lower())]
    t = case.P['t']
    popt, pcov = curve_fit(ff, t, Sy0_opt, p0=[1, guess_freq(t, Sy0_opt), guess_phase(t, Sy0_opt)])
    perr = np.sqrt(np.diag(pcov))
    case.RefFreq = Estimate(popt[1], perr[1])
    case.fit_polarization(var)
    return (popt, perr)


def decoh_meas(case, var):
    dm = case[var][:,2:].std(axis=1)
    var = case[var][:, 1]
    return dm, var

def fit_line(x,y):
    popt, pcov = curve_fit(lambda x, a,b: a + b*x, x, y)
    perr = np.sqrt(np.diag(pcov))
    return (Estimate(popt[0], perr[0]), Estimate(popt[1], perr[1]))

Analysis.fetch_from('../data/DECOH_OBS_20SEC/UNIBEAM/UNOPTSEXT/')
unopt = Analysis('CW', False)
Analysis.fetch_from('../data/DECOH_OBS_20SEC/UNIBEAM/OPTSEXT/')
opt = Analysis('CW', False)


def plot_decoh(case, var='S_Y'):
    dm, sp = decoh_meas(case, var)
    t = case.P['t']
    icpt, slp = fit_line(t, dm)
    f, ax = plt.subplots(1,1)
    ax.plot(t, dm, '--.', label='RMS')
    ax.set_title('Decoherence for '+var)
    ax.set_ylabel('RMS({})'.format(var))
    ax.set_xlabel('t [sec]')
    ax.plot(t, icpt.val + slp.val*t, '-r', label='slp: {:4.2e} +- {:4.2e} (z={:4.2})'.format(slp.val, slp.se, slp.val/slp.se))
    form(ax); ax.legend()
    
