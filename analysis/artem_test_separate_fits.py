from analysis import fit_matrix
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from scipy.stats import linregress

import artem_test as art

font = {'size'   : 16}
plt.rc('font', **font)

sine = art.sine
jitsine = art.jitsine

dat_arr = art.load_case('SEXTVAR/SEXT1')

x = np.column_stack([case.trpray['X'][:,1:] for case in dat_arr])
y = np.column_stack([case.trpray['Y'][:,1:] for case in dat_arr])
it = np.arange(0, x.shape[0]*800, 800)

stunee = np.column_stack([case.stunee[:,1:] for case in dat_arr])
mu0 = stunee['mu0']
nx = stunee['nx']
ny = stunee['ny']
nz = stunee['nz']

meanmu0 = mu0.mean(0)
meannx = nx.mean(0)
meanny = ny.mean(0)
meannz = nz.mean(0)

n = mu0.shape[0]
semu0 = mu0.std(0)/np.sqrt(n)
senx = nx.std(0)/np.sqrt(n)
seny = ny.std(0)/np.sqrt(n)
senz = nz.std(0)/np.sqrt(n)

t = np.column_stack([case['TURN'][:,1:]*1e-6 for case in dat_arr])
sy = np.column_stack([case['S_Y'][:,1:] for case in dat_arr])
sx = np.column_stack([case['S_X'][:,1:] for case in dat_arr])
yoff = np.hstack([case.pray['Y'][1:]*1e3 for case in dat_arr])
ii = np.argsort(yoff)
tbl={}
tbl['Y'] = fit_matrix(t.T, sy.T)
tbl['X'] = fit_matrix(t.T, sx.T)

f, ax = plt.subplots(1,1)
ax.errorbar(yoff[ii], tbl['Y'][ii,0]['f'], yerr=tbl['Y'][ii,1]['f'], fmt='--.')
art.form(ax)
ax.set_xlabel('Y offset [mm]')
ax.set_ylabel('Frequency estimate [Hz]')

f, ax = plt.subplots(1,1)
ax.errorbar(yoff[ii], meanmu0[ii], yerr=semu0[ii], fmt='--.')
art.form(ax)
ax.set_xlabel('Y offset [mm]')
ax.set_ylabel('mean (over time) spin tune')

f, ax = plt.subplots(1,2, sharey=True)
ax[0].errorbar(meanmu0[ii], meannx[ii], yerr=senx[ii], xerr=semu0[ii], fmt='--.')
ax[0].set_xlabel('mean (over time) spin tune', labelpad=20)
ax[0].set_ylabel('mean (over time) nx')
ax[1].errorbar(meanny[ii], meannx[ii], yerr=senx[ii], xerr=seny[ii], fmt='--.')
ax[1].set_xlabel('mean (over time) ny')
art.form(ax[0]); art.form(ax[1])


f, ax = plt.subplots(1,1)
stat='f'
plane='Y'
slp, icpt, r2, pval, stderr = linregress(meanmu0, tbl['Y'][:,0][stat])
ax.errorbar(meanmu0[ii], tbl['Y'][:,0][ii][stat], xerr=semu0[ii], yerr=tbl['Y'][:,1][ii][stat], fmt='--.')
ax.plot(meanmu0, icpt + slp*meanmu0, '-r')
art.form(ax)
ax.set_title('R2 = {:4.2e}'.format(r2))
ax.set_xlabel('mean spin tune')
ax.set_ylabel('{} in the {}-plane'.format(stat.upper(), plane))
