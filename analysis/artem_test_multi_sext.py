
# Artem wants to see the dependence of the Frequency estimate on the centroid offset from the CO
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import artem_test as art
import os, sys
from analysis import Bundle
from copy import deepcopy

font = {'size'   : 16}
plt.rc('font', **font)
form = art.form

OUT_DIR = '../img/Artem/'
STAT_TYPE = [('est', float), ('se', float)]

lin = lambda x, a,b: a+b*x
quad = lambda x, a,b,c:  a + b*x + c*x**2
sine = lambda x, a,f,p: a*np.sin(2*np.pi*f*x + p)
jitsine = lambda x, a0,a1,f0,f1,p: (a0-a1*abs(np.sin(2*np.pi*f1*x)))*np.sin(2*np.pi*f0*x+p)

class CaseContainer:
    def __init__(self, *case_list):
        self.count = len(case_list)
        self.case_list = np.empty(self.count, dtype=object)
        self.fit_parameters = Bundle()
        for i, case in enumerate(case_list):
            self.case_list[i] = art.load_case(DATADIR+'/'+case)
        self.shape = (self.count, self.case_list[0].count)
        
    def fit_polarization(self, var='Y', fit_fun=sine, ini_guess=None):
        guess_keys = ['Ampl', 'Freq', 'Phase'] if ini_guess is None else list(ini_guess.keys())
        
        par_row = dict()
        for key in guess_keys:
            setattr(self.fit_parameters, key, np.empty((self.count, self[0].count), dtype = STAT_TYPE))
            par_row[key] = getattr(self.fit_parameters, key)
            
        for i, case in enumerate(self.case_list):
            case.fit_polarization(var, fit_fun, ini_guess)
            for stat in ['est', 'se']:
                for name in case.fit_parameters:
                    par = getattr(case.fit_parameters, name)
                    par_row[name][stat][i,:] = par[stat]

    def __getitem__(self, index):
        return self.case_list[index]

    def __repr__(self):
        return str(self.shape)

    def cases(self, from_=0, to_=None):
        if to_ is None or to_ > self.count:
            to_= self.count
        for case_id in range(from_, to_):
            yield self[case_id]

def load_sext_settings(cases):
    gsy = np.empty(len(cases), dtype=float)
    for i, case in enumerate(cases):
        gsy[i] = np.loadtxt(case+'SEXTUPOLE.set')
    return gsy
    

if __name__ == '__main__':
    DATADIR = sys.argv[1] if len(sys.argv)>1 else 'SEXTVAR'
    main_path = '../data/ARTEM_TEST/'+DATADIR+'/'
    cases = sorted(list(filter(lambda k: 'SEXT' in k, os.listdir(main_path))), key=lambda x: int(x[4:]))
    cases.pop(0)
    carr = CaseContainer(*cases)
    gsy = load_sext_settings([main_path+case+'/' for case in cases])
    carr.fit_polarization()
    Freq = deepcopy(carr.fit_parameters.Freq)
    pray = carr[0].decoh.pray
    y = pray['Y'][1:]*1e3 # beam's y-offset in mm

    var='nz'
    f, ax = plt.subplots(1,1)
    ax.set_xlabel('Y offset [mm]'); ax.set_ylabel(var)
    for case in range(11):
        ax.plot(y, carr[case].decoh.data[var][1:], '--.', label='{:4.2e}'.format(gsy[case]))
    form(ax); ax.legend()
    
    cids = [0,7,10]
    f,ax = plt.subplots(1,1)
    ax.set_xlabel('Y offset [mm]')
    ax.set_ylabel('Frequency estimate [Hz]')
    for cid in cids:
        ax.errorbar(y, Freq['est'][cid], yerr=Freq['se'][cid], fmt='--.', label='{:4.2e}'.format(gsy[cid]))
    form(ax); ax.legend()
    
    
    
        
    
    
