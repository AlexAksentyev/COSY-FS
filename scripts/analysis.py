import numpy as np
import matplotlib.pyplot as plt
import sys

def get_data(filename):
    with open(filename) as f:
        nray = int(f.readline().strip().split()[-1])
        header = f.readline().strip().split()
        header.pop(0)

    n = len(header)-2 # first two columns in TRPRAY, TRPSPI are iteration number and pid
    dtype = list(zip(header, [int]*2 + [float]*n))
    dat = np.loadtxt(filename,  dtype=dtype, encoding='ASCII')
    nit = int(len(dat)/nray)
    dat.shape = (nit, nray)
    return dat

def analyze(lattice_name, varx='iteration', vary='S_Z'):
    pdat = get_data('../data/'+lattice_name+'/TRPRAY.txt')
    sdat = get_data('../data/'+lattice_name+'/TRPSPI.txt')

    p_names = pdat.dtype.names
    s_names = sdat.dtype.names

    x_flag = [varx in x for x in (p_names, s_names)]
    y_flag = [vary in x for x in (p_names, s_names)]
    x_i = x_flag.index(True)
    y_i = y_flag.index(True)

    xdat = [pdat, sdat][x_i]
    ydat = [pdat, sdat][y_i]
    

    plt.ion()
    plt.figure()
    plt.plot(xdat[varx][:,1:], ydat[vary][:,1:], '.')
    plt.xlabel(varx)
    plt.ylabel(vary)
    plt.title(lattice_name)

    where = '../img/'

    plt.savefig(where+lattice_name+'/'+vary+'_vs_'+varx)

    return pdat, sdat


if __name__ =='__main__':
    latname = sys.argv[1]
    varx = sys.argv[2]
    vary = sys.argv[3]
    ps, spin = analyze(latname, varx, vary)
