# here i model the frequency variation introduced into polarimetry data as a result of
# the spin precession axis motion caused by betatron oscillations

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import sys
from decoh import MU_TYPE
from analysis import Analysis, NLF, LF, Estimate, Bundle, guess_freq, guess_phase
from scipy.optimize import curve_fit
from collections import namedtuple
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess
import inspect
##################################################
# DEFINITIONS
##################################################
HTResult = namedtuple('HTResult', ['score', 'p_val'])
class Centroid:
    def __init__(self, x, y, d):
        self.X = x
        self.Y = y
        self.D = d

    def __add__(self, other):
        return Centroid(self.X + other.X, self.Y + other.Y, self.D + other.D)
    def __sub__(self, other):
        return Centroid(self.X - other.X, self.Y - other.Y, self.D - other.D)
    def __mul__(self, factor):
        return Centroid(self.X * factor, self.Y *factor, self.D *factor)
    def __rmul__(self, factor):
        return self.__mul__(self, factor)
    def __repr__(self):
        return str((self.X, self.Y, self.D))
    def as_tuple(self):
        return (self.X, self.Y, self.D)


class Model:
    def __init__(self, name, data, fitted):
        self.name = name
        self.fitted = fitted
        self.residual = data - fitted

    def append(self, **kwargs):
        for name, val in kwargs.items():
            setattr(self, name, val)
    def __repr__(self):
        return str(list(self.__dict__.keys()))
    
    def residual_vs_fit(self, lowess_frac=.2, fmt='.k'):
        fitted = self.fitted
        res = self.residual
        low_fit = lowess(res, fitted, frac=lowess_frac)
        plt.figure(); plt.title('Run case: '+self.name)
        plt.plot(fitted, res, fmt, label='residuals')
        plt.plot(low_fit[:,0], low_fit[:,1], '-r', label='lowess ({:4.2f})'.format(lowess_frac))
        plt.ylabel('Vertical polarization residual'); plt.xlabel('Model prediction')
        plt.grid(); plt.legend()
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True, useOffset=True)
        # plt.savefig(OUT_DIR+'{}_polarization_residual_vs_fit.png'.format(self.name))
        
    def residual_vs_var(self, var_name, fmt='.k', xlab=None):
        res = self.residual
        var = getattr(self, var_name)
        xlab = var_name if xlab is None else xlab
        plt.figure(); plt.title('Run case: '+self.name)
        plt.plot(var, res, fmt, label='residuals')
        plt.ylabel('Vertical polarization residual'); plt.xlabel(xlab)
        plt.grid(); plt.legend()
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True, useOffset=True)
        # plt.savefig(OUT_DIR+'{}_polarization_residual_vs_{}.png'.format(self.name, var_name))

    
class SMPAnalysis(Analysis):
    def __init__(self, marker, full=True):
        TBL_TYPE = [('turn', int), ('pid', int)] + MU_TYPE
        stune_loader = lambda filename: np.loadtxt(filename, dtype = TBL_TYPE, skiprows=1)
        trdata_loader = lambda filename: LF['TBL'](filename)
        NLF['TRPSPI'] = ('TRPSPI', trdata_loader)
        NLF['TRPRAY'] = ('TRPRAY', trdata_loader)
        Analysis.fetch_from(SMPAnalysis.data_dir)
        loaders = [NLF['PRAY'], NLF['TRPSPI']]
        if full:
            loaders += [NLF['TRPRAY'], ('STUNEE', stune_loader)]
        obj = Analysis.load(marker,  *loaders)
        super().__init__(**obj)
        self.TOF = 1e-6
        self.centroid = Centroid(self.pray['X'].mean(), self.pray['Y'].mean(), self.pray['D'].mean())
        
        self.__comp_pol()

    def __comp_pol(self):
        self.NRAY = self.pray.shape[0]
        self.trpspi.shape = (-1, self.NRAY)
        if getattr(self,'trpray', None) is not None:
            self.trpray.shape = (-1, self.NRAY)
        if getattr(self, 'stunee', None) is not None:
            self.stunee.shape = (-1, self.NRAY)
        ## computing polarization
        SX, SY, SZ = self.trpspi['S_X'], self.trpspi['S_Y'], self.trpspi['S_Z']
        P = np.array([SX.sum(axis=1), SY.sum(axis=1), SZ.sum(axis=1)])
        N = np.linalg.norm(P, axis=0)
        P = P/N
        t = self.trpspi[:, 0]['TURN']*self.TOF
        sy0 = self.trpspi[:, 1]['S_Y'] # spin on the closed orbit
        sx0 = self.trpspi[:, 1]['S_X']
        sz0 = self.trpspi[:, 1]['S_Z']
        self.P = np.array(list(zip(t, sx0, sy0, sz0, *P)), dtype=list(zip(['t', 'Sx0', 'Sy0', 'Sz0', 'X','Y','Z'],[float]*7)))

    def reload(self):
        super().reload()
        self.__comp_pol()
        

    def plot_polarization(self, var='Y', time=True):
        P = self.P
        if time:
            x = P['t']
            xlab = 'time [sec]'
        else:
            x = range(P.shape[0])
            xlab = 'point #'

        ylab = {'Y':'Vertical', 'X':'Horizontal', 'Z':'Longitudinal'}
        refvar = 'S'+var.lower()+'0'
            
        plt.figure(); plt.title('Polarization run case: '+self.marker)
        plt.plot(x, P[var],   '--.r', label='Beam')
        plt.plot(x, P[refvar], '--b',  label='CO particle')
        plt.ylabel(ylab[var]+' polarization'); plt.xlabel(xlab)
        plt.grid(); plt.legend()
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True, useOffset=True)
        plt.savefig(self.data_dir+'{}_polarization.png'.format(self.marker))
        
    def fit_polarization(self, var='Y', fit_fun=lambda x, a,f,p: a*np.sin(2*np.pi*f*x + p), ini_guess=None):
        """ini_guess should be a dictionary of parameter names:parameter guess"""
        P = self.P
        
        if ini_guess is not None and (len(inspect.getargspec(fit_fun).args)-1)!=len(ini_guess):
            raise ValueError('Initial parameter guess dimension is wrong')

        if ini_guess is None:
            Atop = P[var].max()
            p_guess = {'Ampl':Atop, 'Freq':guess_freq(P['t'], P[var]), 'Phase':guess_phase(P['t'], P[var])}
            print('Used initial frequency guess:', p_guess['Freq'])
        else:
            p_guess = ini_guess
        
        popt, pcov = curve_fit(fit_fun, P['t'], P[var], p0=list(p_guess.values()))
        perr = np.sqrt(np.diag(pcov))

        self.model = Model(self.marker, P[var], fit_fun(P['t'], *popt))
        self.model.append(t = self.P['t'], function=fit_fun)
        self.model.append(parameters=Bundle())

        for i, parname in enumerate(p_guess.keys()):
             setattr(self.model.parameters, parname, Estimate(popt[i], perr[i]) )
        
    def power_spectrum(self, cmp='Y', plot=True):
        data = self.P['Y']
        time_step = self.P['t'][1] - self.P['t'][0]
        ps = np.abs(np.fft.fft(data))**2
        freqs = np.fft.fftfreq(data.size, time_step)
        idx = np.argsort(freqs)
        if plot:
            plt.figure(); plt.title('Run case: ', self.marker)
            plt.plot(freqs[idx], ps[idx])
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power')
            plt.grid()
        return np.array(list(zip(freqs, ps)), dtype=[('freq', float), ('pow', float)])
        

    def moving_fit(self, ini_guess=None, frame_size=100, plot=True):
        fit_fun=lambda x, f,p: np.sin(2*np.pi*f*x + p)
        P = self.P[1:]
        if plot:
            plt.figure(); plt.title('Run case: '+self.marker)
            plt.plot(P['t'], P['Y'], '--')
            plt.ylabel('Vertical polarization'); plt.xlabel('time [sec]')
            plt.grid()
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True, useOffset=True)
            
        P.shape = (-1, frame_size)
        esti = np.empty(P.shape[0], dtype=[('est', float), ('se', float)])
        print('Run case:' + self.marker)
        print('Initial parameter guess (f, phase_0):')
        for i, row in enumerate(P):
            t_w = row['t']; py_w = row['Y']
            p_guess = [guess_freq(t_w, py_w), guess_phase(t_w, py_w)]
            print('{}, {}'.format(*p_guess) )
            popt, pcov = curve_fit(fit_fun, t_w, py_w, p0=p_guess)
            perr = np.sqrt(np.diag(pcov))
            esti[i] = popt[0], perr[0]
            if plot:
                plt.plot(t_w, py_w, '--.')

        # if plot:
        #     plt.savefig(OUT_DIR+'{}_moving_fit_frames.png'.format(self.marker))        
                
        self.MF_Freq = esti

    def HT_Freq(self, test_value, dist=lambda score: norm.cdf(score)):
        test_score = (self.Freq.val - test_value)/self.Freq.se
        p_value = 1 - dist(abs(test_score))
        return HTResult(test_score, p_value)
        

## end class


def plot_MF_estimates(cw_case, ccw_case):
    plt.figure(); plt.title('Moving frame fit')
    dat = cw_case.MF_Freq; x = range(dat.shape[0])
    popt, pcov = curve_fit(lambda x, a,b: a + b*x, x, dat['est'], p0=[dat['est'].mean(), 0])
    perr = np.sqrt(np.diag(pcov))
    plt.errorbar(x, dat['est'], yerr=dat['se'], fmt='--.', label='CW')
    plt.plot(x, popt[0] + popt[1]*x, '--r', label='trend: ({0:4.2e} +- {2:4.2e}, {1:4.2e} +- {3:4.2e})'.format(*popt, *perr))
    dat = ccw_case.MF_Freq; x = range(dat.shape[0])
    plt.errorbar(x, dat['est'], yerr=dat['se'], fmt='--.', label='CCW')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=True, useMathText=True)
    plt.ylabel('Frequency estimate [Hz]')
    plt.xlabel('Point #')
    plt.legend()
    # plt.savefig(OUT_DIR+'moving_frame_freqs.png')

def plot_beam_hists(cw_case, ccw_case, out_dir='../img/dump/'):
    xcw = cw_case.pray['X']*1e3; xccw = ccw_case.pray['X']*1e3
    ycw = cw_case.pray['Y']*1e3; yccw = ccw_case.pray['Y']*1e3
    dcw = cw_case.pray['D'];     dccw = ccw_case.pray['D']
    dx = xcw.mean() - xccw.mean()
    dy = ycw.mean() - yccw.mean()
    dd = dcw.mean() - dccw.mean()
    
    f, ax = plt.subplots(2,1, sharex=True)
    ax[0].set_xlabel('X [mm]'); ax[0].set_ylabel('Count')
    ax[1].set_xlabel('Y [mm]'); ax[1].set_ylabel('Count')

    ax[0].set_title('Mean difference: {:4.2e} [mm]'. format(dx))
    ax[0].hist(xcw, color='b', histtype='step', label='CW')
    ax[0].axvline(x=xcw.mean(), color='b', linestyle='--')
    ax[0].hist(xccw, color='r', histtype='step', label='CCW')
    ax[0].axvline(x=xccw.mean(), color='r', linestyle='--')
    ax[0].legend()
    
    ax[1].set_title('Mean difference: {:4.2e} [mm]'. format(dy))
    ax[1].hist(ycw, color='b', histtype='step', label='CW')
    ax[1].axvline(x=ycw.mean(), color='b', linestyle='--')
    ax[1].hist(yccw, color='r', histtype='step', label='CCW')
    ax[1].axvline(x=yccw.mean(), color='r', linestyle='--')
    ax[1].legend()
    plt.savefig(out_dir+'beam_histograms_XY.png')

    f1, ax1 = plt.subplots(1,1)
    ax1.set_title('Mean difference: {:4.2e}'. format(dd))
    ax1.set_xlabel('D'); ax1.set_ylabel('Count')
    ax1.hist(dcw, color='b', histtype='step', label='CW')
    ax1.axvline(x=dcw.mean(), color='b', linestyle='--')
    ax1.hist(dccw, color='r', histtype='step', label='CCW')
    ax1.axvline(x=dccw.mean(), color='r', linestyle='--')
    ax1.legend()
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    plt.savefig(out_dir+'beam_histograms_D.png')
    
    
##################################################
## OLD CODE ##
##################################################
def plot_three(case, pids, freq='x'):
    ncmp = 'n'+freq
    w = lambda tbl: 2*np.pi/case.TOF*tbl['mu0']*abs(tbl[ncmp])
    stunee = case.stunee
    pray = case.pray
    f, ax = plt.subplots(3,1)
    for lab, pid in pids.items():
        n = stunee[:, pid][ncmp]
        mu0 = stunee[:, pid]['mu0']
        W = w(stunee[:, pid])
        lab = 'offset: ({:4.2e}, {:4.2e})'.format(*pray[pid][['X', 'Y']])
        ax[0].plot(mu0, '--.', label=lab)
        ax[0].set_ylabel('spin tune'); ax[0].legend()
        ax[1].plot(n, '--.', label=lab)
        ax[1].set_ylabel('n_'+freq); ax[1].set_xlabel('turn #'); ax[1].legend()
        ax[0].ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
        ax[1].ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
        ax[2].hist(W, label=lab, histtype='step')
        ax[2].set_xlabel('W_{} [rad/sec]'.format(freq)); ax[2].set_ylabel('count'); ax[2].legend()
        ax[2].ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
    f.savefig(OUT_DIR+'spin_tune_three.png')

##################################################
OUT_DIR = '../../EDM/Reports/PhD/img/spin_axis_motion/presentation/'
def load_data(indir):
    print('LOADING DATA')

if __name__ == '__main__':

    INDIR = '../../data/SMP/{}/'.format(sys.argv[1]) if len(sys.argv)>1 else '../data/SMP/'
    SMPAnalysis.fetch_from(INDIR)
    cw_run =  SMPAnalysis('CW')
    # ccw_run = SMPAnalysis('CCW')

    cw_run.fit_polarization(); cw_run.Fits.residual_vs_var('t')
    # ccw_run.fit_polarization()

    ### single particle spin tune and n bar
    if False:
        i = 0
        i_x = np.arange(2, cw_run.NRAY, 4); i_y = i_x + 1;
        i_d = i_y + 1; i_xy = i_d + 1
        pids = {'A': i_xy[15], 'B':i_xy[200], 'C':i_xy[799]}

        plot_three(cw_run, pids)
    ###
