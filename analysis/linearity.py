from analysis import Analysis, NLF
from CALIB.calib_danf import MU_TYPE
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
TOF = 1e-6
Analysis.fetch_from('../data/LINEARITY/')
DTYPE = [('case', int)] + MU_TYPE

def load_gauss(analysis):
    mrkrs=[int(fname.replace(':', '.').split('.')[1]) for fname in data.list_files('GAUSS')]
    mrkrs.sort()
    gauss = np.empty((len(mrkrs), 32))
    for i, mrkr in enumerate(mrkrs):
        gauss[i] = np.loadtxt(analysis.data_dir+'GAUSS:{}.in'.format(mrkr))/180*np.pi
    return gauss

NLF['STUNE0'] = ('STUNE0', lambda name: NLF['MU'][1](name, DTYPE))
data = Analysis.load('CW', NLF['STUNE0'])
data.gauss = load_gauss(data)

mean_tilt = data.gauss.mean(axis=1)
mu0 = data.stune0['mu0']
ftr = np.pi*2*mu0/TOF
nx = data.stune0['nx']; ny = data.stune0['ny']

std_x = 1# mean_tilt.std()
fmt = '--.'
x_lab = 'mean tilt angle [rad]'

f, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(mean_tilt/std_x, nx, fmt, label='nx')
ax[0].plot(mean_tilt/std_x, ny, fmt, label='ny')
ax[0].set_xlabel(x_lab)
ax[0].set_ylabel('n-bar')
ax[0].set_title('Spin precession axis')

ax[1].plot(mean_tilt/std_x, ftr*nx, fmt, label='Wx')
ax[1].plot(mean_tilt/std_x, ftr*ny, fmt, label='Wy')
ax[1].set_xlabel(x_lab)
ax[1].set_ylabel('W [rad/sec]')
ax[1].set_title('Precession frequency')
ax[1].ticklabel_format(style='sci', axis='both', scilimits=(0,0))

for i in range(2):
    ax[i].grid(); ax[i].legend()


