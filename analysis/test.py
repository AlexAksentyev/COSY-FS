import lmfit
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from SMP.spin_motion_perturbation import guess_freq, guess_phase, SMPAnalysis
import artem_test as art
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sine = art.sine
jitsine = art.jitsine

SMPAnalysis.fetch_from('../data/ARTEM_TEST/SEXTVAR/SEXT1')

def relLik(model0, model1):
    return np.exp((model0.aic - model1.aic)/2)

dat = SMPAnalysis('CW1')

n = dat.trpspi.shape[0]
sx = dat.trpspi['S_X'][:, 28] #+ np.random.normal(0,1e-2,n)
sy = dat.trpspi['S_Y'][:, 28] #+ np.random.normal(0,1e-2,n)
Py = dat.P['Y'] #+ np.random.normal(0,1e-2,n)
t = dat.trpspi['TURN'][:, 28]*1e-6

syf0g = guess_freq(t,sy)
syp0g = guess_phase(t,sy)

smodel = lmfit.Model(sine)
jsmodel = lmfit.Model(jitsine)
spars = smodel.make_params(a=1,f=syf0g,p=syp0g)
spars['a'].set(max=1, min=.9)
jspars = jsmodel.make_params(a0=1, a1=0,f0=syf0g, f1=syf0g,p=syp0g)
jspars['a0'].set(max=1)

sres = smodel.fit(sy,spars,x=t)
jsres = jsmodel.fit(sy,jspars,x=t)

sres.plot_fit()



