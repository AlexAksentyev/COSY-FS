## analysis script for the making of my seminar presentation
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from SMP.spin_motion_perturbation import guess_freq, guess_phase, SMPAnalysis
import artem_test as art
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

font = {'size'   : 14}
plt.rc('font', **font)

SMPAnalysis.fetch_from('../data/ARTEM_TEST/SEXTVAR/SEXT1')

dat = SMPAnalysis('CW1')

f, ax = plt.subplots(1,1)
pidmin = 28
ymin = dat.pray[pidmin]['Y'] * 1e3
pidmax = 24
ymax = dat.pray[pidmax]['Y'] * 1e3
ax.plot(dat['T'][:,pidmin], dat['D'][:,pidmin], '--.',label = 'y-offset: {:4.2f} [mm]'.format(ymin))
ax.plot(dat['T'][:,pidmax], dat['D'][:,pidmax], '--.',label = 'y-offset: {:4.2f} [mm]'.format(ymax))
ax.set_xlabel('Phase difference'); ax.set_ylabel('(K-K0)/K0')
ax.legend(); ax.grid()
art.form(ax)


dat.fit_polarization()
plot_pacf(dat.model.residual, lags=10)
