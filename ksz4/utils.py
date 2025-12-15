import os
import errno
import yaml
import falafel.utils as futils
import healpy as hp
import numpy as np
from pixell import curvedsky

def get_disable_mpi():
    try:
        disable_mpi_env = os.environ['DISABLE_MPI']
        disable_mpi = True if disable_mpi_env.lower().strip() == "true" else False
    except:
        disable_mpi = False
    return disable_mpi

def safe_mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise(e)

config = futils.config
def get_cmb_alm_unlensed(i,iset,path=config['signal_path']):
    sstr = str(iset).zfill(2)
    istr = str(i).zfill(5)
    fname = path + "fullskyUnlensedCMB_alm_set%s_%s.fits" % (sstr,istr)
    return hp.read_alm(fname,hdu=(1,2,3))


def get_cl_smooth(alm1, alm2=None, n=5):
    if alm2 is None:
        alm2=alm1
    cl = curvedsky.alm2cl(alm1, alm2)
    window = np.ones(int(n))/float(n)
    cl_out = np.convolve(cl, window, mode="same")
    cl_out[-n:] = cl[-n:]
    return cl_out


class ClBinner(object):
    """
    Class for binning Cls in equal
    width bins, weighting by 2L+1
    """
    def __init__(self, lmin=100, lmax=1000, nbin=20, log=False):
        self.lmin=lmin
        self.lmax=lmax
        self.nbin=nbin
        if log:
            log_bin_lims = np.linspace(
                np.log(lmin), np.log(lmax), nbin+1)
            #need to make this integers
            bin_lims = np.ceil(np.exp(log_bin_lims))
            #remove duplicates
            bin_lims = np.unique(bin_lims).astype(int)
            self.bin_lims = bin_lims
            self.nbin=len(self.bin_lims)-1
            log_bin_lims = np.log(self.bin_lims)
            log_bin_mids = 0.5*(log_bin_lims[:-1]+log_bin_lims[1:])
            self.bin_mids = np.exp(log_bin_mids)
        else:
            self.bin_lims = np.ceil(np.linspace(
                self.lmin, self.lmax+1, self.nbin+1
            )).astype(int)
            self.bin_mids = 0.5*(self.bin_lims[:-1]
                                 +self.bin_lims[1:])
        self.deltal = np.diff(self.bin_lims)
        
    def bin_errorbars(self, sigmas):
        #Var(binned_cl) = \sum_{L1 < L < L2} w_L^2 sigma_L^2 / (\sum_{L1 < L < L2} w_L^2)
        L = np.arange(len(sigmas)).astype(int)
        var_binned_cl = np.zeros(self.nbin)
        w = 2*L+1
        for i in range(self.nbin):
            use = (L>=self.bin_lims[i])*(L<self.bin_lims[i+1])
            var_binned_cl[i] = (w[use]**2 * sigmas[use]**2).sum() / (w[use]**2).sum()
        return np.sqrt(var_binned_cl)
    
    def __call__(self, cl):
        L = np.arange(len(cl)).astype(int)
        w = 2*L+1
        cl_binned = np.zeros(self.nbin)
        for i in range(self.nbin):
            use = (L>=self.bin_lims[i])*(L<self.bin_lims[i+1])
            cl_binned[i] = np.average(cl[use], weights=w[use])
        return cl_binned
