import SMP.spin_motion_perturbation as smp
import artem_test as art
from analysis import Estimate
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.ion()


NRAY = 1 # ensemble (beam) size

F0 = 17.3 # spin precession frequency (nominal)
FS = F0*15 # sampling frequency
T = 1

f0 = np.random.normal(F0, 0, NRAY) # ensemble spin precession frequencies (decoherence)
PREC_AMPL = 0*1e-6 # SPA variation amplitude
f1 = np.random.normal(173.53, 0, NRAY) # SPA precession frequencies
p1 = 0*np.linspace(-1e-6, 1e-6, NRAY) # SPA phases

t = np.linspace(0, T, int(T*FS)).repeat(NRAY).reshape(-1, NRAY)
sy = np.sin(2*np.pi*f0*t)
a = 1-PREC_AMPL*abs(np.sin(2*np.pi*f1*t + p1)) # abs b/c spin cannot exceed 1,
                                        # if the SPA tilts, S projection decreases
err = np.random.normal(0, 1e-4, t.shape[0])
syp = a*sy

Py = syp.sum(axis=1)
Py = Py/Py.max() + err*0
plt.plot(t[:,0], Py, '--.')

popt, perr = art.fit_sine(t[:,0], Py)
res, fits = art.res_fit(t[:,0], Py)
print('*** Residual normality test', stats.normaltest(res))

Freq = Estimate(popt[1], perr[1])
Ampl = Estimate(popt[0], perr[0])
Phase = Estimate(popt[2], perr[2])
print('*** Freq deviation', Freq - Estimate(F0,0))
print('*** Ampl deviation', Ampl - Estimate(1,0))
print('*** Phase deviation', Phase - Estimate(0,0))
