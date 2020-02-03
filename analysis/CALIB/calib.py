from analysis import Analysis, Bundle
import os; import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from calib_util import _w_fit

plt.ioff()

PI  = np.pi
TOF = 1e-6
HOME_DIR = os.environ['HOME']+'/REPOS/COSYINF/'

W_STAT = '3d'

# **************************************************
def load_data(X='all', S='all'):
    from analysis import NLF, LF
    X = '[0-9]*' if X=='all' else X
    S = '[0-9]*' if S=='all' else S
    
    cw_markers = sort_markers(Analysis.list_markers('CW.{}.{}\.dat'.format(X,S)))
    ccw_markers= sort_markers(Analysis.list_markers('CCW.{}.{}\.dat'.format(X,S)))

    acw = []; accw = []
    for marker in cw_markers:
        acw.append(Analysis.load(marker, NLF['TRPSPI'], NLF['TRPRAY'], NLF['PRAY'], NLF['PSPI']))
        acw[-1].gauss = LF['COL']('{}GAUSS{}.in'.format(acw[-1].data_dir, marker[2:]))
    for marker in ccw_markers:
        accw.append(Analysis.load(marker, NLF['TRPSPI'], NLF['TRPRAY'], NLF['PRAY'], NLF['PSPI']))
    for i, a in enumerate(accw):
        a.gauss = np.flip(acw[i].gauss, 0)
    return acw, accw

def sort_markers(mrks_lst):
    mrks = np.array([tuple(e.replace('X', ':').replace('S',':').split(':')) for e in mrks_lst],
                    dtype=[('mrk', object), ('X', int), ('S', int)])
    mrks.sort(order=['X', 'S'])
    return np.array(['{}X{}S{}'.format(*e) for e in mrks])
# **************************************************
def form_descr(data_list):
    descr = Bundle();
    descr.W, descr.stune, descr.tilt, descr.Wcmp = join(data_list)
    return descr

def join(lst):
    from analysis import STAT_TYPE, omega, stune
    n_lst = len(lst); nray = lst[0].pray.shape[0]
    w = np.empty((n_lst, nray, 3), dtype=STAT_TYPE)
    w_comp = np.empty((n_lst, nray, 3), dtype=STAT_TYPE)
    st = np.empty((n_lst, nray, 4), dtype=STAT_TYPE)
    tilt = np.empty((n_lst, lst[0].gauss.shape[0]))
    for i, a in enumerate(lst):
        w[i] = omega(a)
        w_comp[i] = _w_fit(a, out_file)
        st[i] = stune(w[i])
        tilt[i] = a.gauss
    
    return w, st, tilt, w_comp

def form_stats(label):
    if label == 'fit': # computes the spin COMPONENT oscillation frequencies via fitting
        IX = 1; IY = 0 # Wx is the Sy component (index 1) oscillation frequency, Wy --- Sx (index 0)
        Wcw = CW.Wcmp; Wccw = CCW.Wcmp
    else: # computes the spin VECTOR osccillation frequency vector via w = S x S_dot
        IX = 0; IY = 1 # Wx is the 1st component, Wy the second, of (Wx, Wy, Wz)
        Wcw = CW.W; Wccw = CCW.W

    S1 = Wcw['est'][:, :, IY] - Wccw['est'][:, :, IY]
    S1std = np.sqrt(Wcw['std'][:, :, IY]**2 +Wccw['std'][:, :, IY]**2)
    S2 = Wcw['est'][:, :, IX] - Wccw['est'][:, :, IX]
    S2std = np.sqrt(Wcw['std'][:, :, IX]**2 + Wccw['std'][:, :, IX]**2)
    return S1, S1std, S2, S2std

# **************************************************
def main_plot(pids=[0], case_tbl=None):
    if case_tbl is None:
        case_tbl = [range(N_ROW)]
    plot_func = lambda x,y,xerr, yerr, fmt, lbl: plt.plot(x,y, fmt, label=lbl)
    f = plt.figure()
    for cases in case_tbl:
        for pid in pids:
            plot_func(S1[cases][:, pid], S2[cases][:, pid],
                        S1std[cases][:, pid], S2std[cases][:, pid],
                        '.', pid)
        plt.xlabel('Delta Wy'); plt.ylabel('Delta Wx')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.grid(); plt.legend()
        plt.title(SETUP)
    return f


        
if __name__ == '__main__':
    WHERE = sys.argv[1]
    SETUP = sys.argv[2] if len(sys.argv)>2 else 'BNL'
    X_RNG = sys.argv[3] if len(sys.argv)>3 else 'all'
    S_RNG = sys.argv[4] if len(sys.argv)>4 else 'all'

    DATA_DIR = HOME_DIR+"data/"+SETUP+"/calib/{}/".format(WHERE)
    Analysis.fetch_from(DATA_DIR)
    
    REPORT_FILE = PdfPages(HOME_DIR+'report/calib_report({}_{}).pdf'.format(SETUP, WHERE))

    acw, accw = load_data(X_RNG, S_RNG)
    print('DATA LOADED')

    CASE_TBL = np.array([range(len(acw))])
    mrk = Analysis.list_markers('CW.{}.{}\.dat'.format(X_RNG,S_RNG))
    N_ROW = len(np.unique([e.replace("X",":").replace("S",":").split(":")[1]for e in mrk]))
    CASE_TBL.shape = (N_ROW, -1)
    CASE_TBL = CASE_TBL.T

    out_file = PdfPages(HOME_DIR+'/report/fit_CW_report.pdf')
    CW  = form_descr(acw)
    out_file.close()
    out_file = PdfPages(HOME_DIR+'/report/fit_CCW_report.pdf')
    CCW = form_descr(accw)
    out_file.close()
    print('DESCRIPTIONS FORMED')

    # difference statistics (CW - CCW)
    S1, S1std, S2, S2std = form_stats(W_STAT)

    pray = acw[0].pray
    pspi = acw[0].pspi
    ix = np.arange(1,22,2); iy = ix+1
    isx = np.arange(23,44,2); isy = isx+1
    x = pray['X'][ix]; y = pray['Y'][iy]
    sx = pspi['S_X'][isx]; sy = pspi['S_Y'][isy]

    fmt='--.'
    case=0
    f, ax = plt.subplots(2,2, sharex='col', sharey='row')
    ax[0,0].set_ylabel('S1'); ax[1,0].set_xlabel('SX')
    ax[1,0].set_ylabel('S2'); ax[1,1].set_xlabel('SY')
    ax[0,0].errorbar(x, S1[case,isx], yerr=S1std[case, isx], fmt=fmt)
    ax[0,1].errorbar(y, S1[case,isy], yerr=S1std[case, isy], fmt=fmt)
    ax[1,0].errorbar(x, S2[case,isx], yerr=S2std[case, isx], fmt=fmt)
    ax[1,1].errorbar(y, S2[case,isy], yerr=S2std[case, isy], fmt=fmt)
    for i in range(2):
        for j in range(2):
            ax[i,j].grid()


    dat0 =  CW.W[case, iy, 1]
    dat1 = CCW.W[case, iy, 1]
    plt.figure()
    plt.errorbar(x, dat0['est'], yerr=dat0['std'], fmt=fmt)
    plt.errorbar(x, dat1['est'], yerr=dat1['std'], fmt=fmt)

    f, ax = plt.subplots(2,1)
    ax[0].errorbar(sx, CW.Wcmp[case, isx, 1]['est'], yerr=0*CW.Wcmp[case, isx, 1]['std'], fmt=fmt)
    ax[0].set_ylabel('Est(Wx)'); ax[0].set_xlabel('SX')
    ax[1].errorbar(sy, CW.Wcmp[case, isy, 1]['est'], yerr=0*CW.Wcmp[case, isy, 1]['std'], fmt=fmt)
    ax[1].set_ylabel('Est(Wx)'); ax[1].set_xlabel('SY')

    f, ax = plt.subplots(3,1)
    ax[0].set_ylabel('S_X'); ax[0].plot(acw[case]['S_X'][:, isx[:3]])
    ax[1].set_ylabel('S_Y'); ax[1].plot(acw[case]['S_Y'][:, isx[:3]])
    ax[2].set_ylabel('S_Z'); ax[2].plot(acw[case]['S_Z'][:, isx[:3]])

    plt.ion()
    print('RESULTS PLOTTED')
    plt.show()
