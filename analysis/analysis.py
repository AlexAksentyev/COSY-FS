#!/usr/bin/env python3.6

import numpy as np
import matplotlib.pyplot as plt
import os, re
from matplotlib.backends.backend_pdf import PdfPages
from copy import deepcopy
import sympy

def _separate(name):
    try:
        name, ext = name.split('.')
    except ValueError:
        ext = 'dat'
    return name, ext    

def load_tr_data(filename):
    with open(filename) as f:
        line0 = f.readline().strip().split()
        for word in line0:
            try:
                nray = int(word)
            except:
                pass
        header = f.readline().strip().split()
        header.pop(0)

    int_col = [e.islower() for e in header] # integer column names are written in lower case
    nint = sum(int_col); nfloat = len(header) - nint
    dtype = list(zip(header, [int]*nint + [float]*nfloat))
    dat = np.loadtxt(filename,  dtype=dtype)#, encoding='ASCII') #encoding doesn't exist for np 1.13
    nit = int(len(dat)/nray)
    dat.shape = (nit, nray)
    return dat[:, :]

def load_tbl(filename, dtype=None):
    if dtype is None:
        with open(filename) as f:
            header = f.readline().strip().split()
            if header[0]=='#':
                header.pop(0) # first symbol is #
        dtype = list(zip(header, [float]*len(header)))
    dat = np.loadtxt(filename, dtype, skiprows=1)
    return dat

LF = dict(
    TR = load_tr_data,
    TBL = load_tbl,
    COL = lambda filename: np.loadtxt(filename)
)

NLF = dict(
    MU=('MU', LF['TBL']),
    GAUSS=('GAUSS.in', LF['COL']),
    PRAY=('PRAY', lambda fname: LF['TR'](fname).flatten()),
    PSPI=('PSPI', lambda fname: LF['TR'](fname).flatten()),
    TRPRAY=('TRPRAY', LF['TR']),
    TRPSPI=('TRPSPI', LF['TR']),
    STUNEE=('STUNEE', LF['TBL'])
)

class Estimate:
    def __init__(self, val, se):
        self.val = val
        self.se = se       
    def __sub__(self, other):
        est = self.val - other.val
        se = np.sqrt(self.se**2 + other.se**2)
        return Estimate(est, se)
    def __add__(self, other):
        est = self.val + other.val
        se = np.sqrt(self.se**2 + other.se**2)
        return Estimate(est, se)
    def __mul__(self, factor):
        return Estimate(factor*self.val, factor*self.se)
    def __rmul__(self, factor):
        return self.__mul__(factor)
    def __repr__(self):
        from scipy.stats import norm
        z = self.val/self.se
        p = (1-norm.cdf(abs(z)))*100
        return str('\nVal = {:4.2e} +- {:4.2e} \nZ = {:4.2f} \nP(0|Norm) = {:4.2f}%'.format(self.val, self.se, z, p))

class Bundle(dict):
    """FROM: 
    http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
    """
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__ = self

    def __deepcopy__(self, memo):
        res = self.__class__(**self.__dict__)
        return res

    def __repr__(self):
        return str(list(self.keys()))

class Analysis(Bundle):
    _mark_sep = ':'
    
    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def fetch_from(cls, directory):
        if directory[-1]!="/": directory += "/"
        cls.data_dir = directory
        cls.fname_list = os.listdir(cls.data_dir)

    @classmethod
    def list_files(cls, regex):
        cls.fname_list = os.listdir(cls.data_dir)
        return list(filter(re.compile(regex).search, cls.fname_list))
    @classmethod
    def list_markers(cls, regex):
        filenames = cls.list_files(cls._mark_sep+regex)
        return np.unique([e.replace(cls._mark_sep, '.').split('.')[1] for e in filenames])

    @classmethod
    def load(cls, modifier, *name_loader):
        sep = cls._mark_sep if modifier!='' else ''
        filename = lambda name='[XXX]', ext='dat': cls.data_dir+'{}{}{}.{}'.format(name, sep, modifier, ext)
        data = dict()
        print('Data for: ', modifier, '({})'.format(cls.data_dir))
        for name, loader in name_loader:
            name, ext = _separate(name)
            print('*** Loading ', name)
            data[name.lower()] = loader(filename(name, ext))
        obj = cls(**data)
        obj.data_dir = cls.data_dir
        obj.filename = filename
        obj.marker = modifier
        obj._name_loader = name_loader
        return obj

    def append(self, *name_loader):
        for name, loader in name_loader:
            name, ext = _separate(name)
            self[name.lower()] = loader(self.filename(name, ext))

    def reload(self):
        self.append(*self._name_loader)

    def __getitem__(self, name):
        data = getattr(self, 'trpspi', None) if name[:2]=='S_' else getattr(self, 'trpray', None)
        return data[name]

class DAVEC:
    VARS = ['X','A','Y','B','T','D']
    def __init__(self, path):
        X,A,Y,B,T,D = sympy.symbols(self.VARS)
        self._dtype = list(zip(['i', 'coef', 'ord'] + self.VARS, [int]*9))
        self._dtype[1] = ('coef', float)
        self._data = np.loadtxt(path, skiprows=1,  dtype=self._dtype, comments='-----')
        self.const = self._data[0]['coef']
        cc = self._data['coef']
        e = {}
        for var in self.VARS:
            e[var] = self._data[var]
        expr = cc*(X**e['X'] * A**e['A'] * Y**e['Y'] * B**e['B'] * T**e['T'] * D**e['D'])
        self.coefs = cc
        self.expr = expr
        self.poly = sympy.poly(expr.sum()) # wanted to improve this with list and Poly, but
        # "list representation is not supported," what?
        
    def __call__(self, ps_arr):
        # vals = np.array([self.poly(*v) for v in ps_arr]) # for some reason this doesn't work properly
        vals = np.array([self.poly.eval(dict(zip(ps_arr.dtype.names, v))) for v in ps_arr]) # this does
        return vals

    def __sub__(self, other):
        return self.poly.sub(other.poly)

    def __add__(self, other):
        return self.poly.add(other.poly)

def load_davecs(path, marker='FULL'):
    print('DAVECS PATH: ', path)
    nu = DAVEC(path+'/MU:{}.dat'.format(marker))
    nx = DAVEC(path+'/NBAR1:{}.dat'.format(marker))
    ny = DAVEC(path+'/NBAR2:{}.dat'.format(marker))
    nz = DAVEC(path+'/NBAR3:{}.dat'.format(marker))
    return dict(zip(['mu0','nx','ny','nz'], [nu, nx, ny, nz]))

## ******** GENERAL USE FUNCTIONS ********** ##
import lmfit
sine = lambda x, a,f,p: a*np.sin(2*np.pi*f*x + p)

def guess_freq(time, signal): # estimating the initial frequency guess
    zci = np.where(np.diff(np.sign(signal)))[0] # find indexes of signal zeroes
    delta_phase = np.pi*(len(zci)-1)
    delta_t = time[zci][-1]-time[zci][0]
    guess = delta_phase/delta_t/2/np.pi
    return guess

def guess_phase(time, sine):
    ds = sine[1]-sine[0]
    dt = time[1]-time[0]
    sg = np.sign(ds/dt)
    phase0 = np.arcsin(sine[0]) if sg>0 else np.pi-np.arcsin(sine[0])
    return phase0

def fit_arrays(x,y,fit_fun=sine, ini_guess=None):
    model = lmfit.Model(fit_fun)
    if ini_guess is None:
        atop = y.max()
        fg = guess_freq(x,y)
        pg = guess_phase(x,y)
        ini_guess = dict(zip(model.param_names, [atop,fg,pg]))
    elif model.param_names!=list(ini_guess.keys()):
        raise ValueError('Incorrect initial guess spec!')

    pars = model.make_params(**ini_guess)
    indep = {model.independent_vars[0]:x}
    result = model.fit(y,pars,**indep)
    fit_pars = result.params
    npars = len(fit_pars)
    tbl = np.empty(2, dtype=list(zip(fit_pars.keys(), [float]*npars)))
    for name, par in fit_pars.items():
        est = par.value; se = par.stderr
        tbl[name] = est, se
    return tbl
    

def fit_matrix(xdat, ydat, fit_fun=sine, ini_guess=None):
    nrows = ydat.shape[0]
    model = lmfit.Model(fit_fun)
    pnames = model.param_names
    npars = len(pnames)
    tbl = np.empty((nrows, 2), dtype=list(zip(pnames, [float]*npars)))
    for i, yrow in enumerate(ydat):
        xrow = xdat[i]
        tbl[i] = fit_arrays(xrow, yrow, fit_fun, ini_guess)
    return tbl
