import numpy as np
import healpy as hp
import pytempura
from pytempura import norm_general, noise_spec
import os
from falafel import utils as futils, qe
from orphics import maps
from pixell import lensing, curvedsky
import matplotlib.pyplot as plt
from os.path import join as opj
import pickle
import solenspipe
from cmbsky import norm_qtt_asym, norm_xtt_asym
from scipy.signal import savgol_filter

try:
    WEBSKY_DIR=os.environ["WEBSKY_DIR"]
except KeyError:
    WEBSKY_DIR="/global/project/projectdirs/act/data/maccrann/websky"
try:
    SEHGAL_DIR=os.environ["SEHGAL_DIR"]
except KeyError:
    SEHGAL_DIR="/global/project/projectdirs/act/data/maccrann/sehgal"

#This can do strange things at low l
"""
def get_cl_fg_smooth(alms, alm2=None):
    cl = curvedsky.alm2cl(alms, alm2=alm2)
    l = np.arange(len(cl))
    d = l*(l+1)*cl
    #smooth with savgol
    d_smooth = savgol_filter(d, 5, 2)
    #if there's still negative values, set them
    #to zero
    d_smooth[d_smooth<0.] = 0.
    return np.where(
        l>0,d_smooth/l/(l+1),0.)    
"""

def get_cl_smooth(alm1, alm2=None, n=5):
    if alm2 is None:
        alm2=alm1
    cl = curvedsky.alm2cl(alm1, alm2)
    window = np.ones(int(n))/float(n)
    cl_out = np.convolve(cl, window, mode="same")
    cl_out[-n:] = cl[-n:]
    return cl_out

def filter_T(T_alm, cltot, lmin, lmax):
    """                                                                                                                                                                                                    
    filter by 1/cltot within lmin<=l<=lmax                                                                                                                                                                 
    return zero otherwise                                                                                                                                                                                  
    """
    mlmax=qe.get_mlmax(T_alm)
    """
    print("mlmax:",mlmax)
    print("lmin:", lmin)
    print("lmax:", lmax)
    """
    filt = np.zeros_like(cltot)
    ls = np.arange(filt.size)
    assert lmax<=ls.max()
    assert lmin<lmax
    filt[2:] = 1./cltot[2:]
    filt[ls<lmin] = 0.
    filt[ls>lmax] = 0.
    return curvedsky.almxfl(T_alm.copy(), filt)

def dummy_teb(alms):
    return [alms, np.zeros_like(alms), np.zeros_like(alms)]

def norm_qtt_asym(est,lmax,glmin,glmax,llmin,llmax,
                   rlmax,TT,OCTG,OCTL,gtype='',profile=None):
    if ((est=='src') and (profile is not None)):
        norm = norm_general.qtt_asym(
            est,lmax,glmin,glmax,llmin,llmax,
            rlmax, TT, OCTG/(profile[:glmax+1]**2),
            OCTL/(profile[:llmax+1]**2), 
            gtype=gtype)
        return (norm[0]*profile**2, norm[1]*profile**2)
    else:
        return norm_general.qtt_asym(
            est,lmax,glmin,glmax,llmin,llmax,
                   rlmax,TT,OCTG,OCTL,gtype=gtype)
    
def noise_xtt_asym(est, mlmax, lmin, lmax, wLA, wGB, wLC, wGD,
                   cltot_AC, cltot_BD, cltot_AD, cltot_BC, profile=None):
    if ((est=="srclens")) and (profile is not None):
        return noise_spec.xtt_asym("srclens", mlmax,lmin,lmax,
            wLA, wGB, wLC, wGD,
            cltot_AC[:lmax+1], cltot_BD[:lmax+1], cltot_AD[:lmax+1], cltot_BC[:lmax+1])/profile
    elif ((est=="lenssrc")) and (profile is not None):
        return noise_spec.xtt_asym("lenssrc", mlmax,lmin,lmax,
            wLA, wGB, wLC, wGD,
            cltot_AC[:lmax+1], cltot_BD[:lmax+1], cltot_AD[:lmax+1], cltot_BC[:lmax+1])/profile 
    
    elif est in ["Ksrc","srcK"]:
        n = noise_spec.qtt_asym(
            "src", mlmax,lmin,lmax,
            wLA, wGB, wLC, wGD,
            cltot_AC[:lmax+1], cltot_BD[:lmax+1], cltot_AD[:lmax+1], cltot_BC[:lmax+1])[0]/profile
        assert len(n)==mlmax+1
        return n
    else:
        return noise_spec.xtt_asym(est, mlmax,lmin,lmax,
            wLA, wGB, wLC, wGD,
            cltot_AC[:lmax+1], cltot_BD[:lmax+1], cltot_AD[:lmax+1], cltot_BC[:lmax+1])
        
    
def norm_xtt_asym(est,lmax,glmin,glmax,llmin,llmax,rlmax,
                   TT,OCTG,OCTL,gtype='',profile=None):

    if ((est=="lenssrc") and (profile is not None)):
        r = norm_general.xtt_asym(est,lmax,glmin,glmax,llmin,llmax,rlmax,
                                  TT, OCTG/(profile[:llmax+1]), OCTL/(profile[:llmax+1]), gtype=gtype)
        return r/profile
    
    elif ((est=="srclens") and (profile is not None)):
        r = norm_general.xtt_asym(est,lmax,glmin,glmax,llmin,llmax,rlmax,
                                  TT, OCTG/(profile[:glmax+1]), OCTL/(profile[:glmax+1]), gtype=gtype)
        return r/profile
    
    elif est=="srcK":
        print("!!!!!!!!!!!!!!")
        print("I'm not too sure about these src-K cross responses...")
        print("need to run numerical tests")
        inv_r = (norm_general.qtt_asym(
            "src",lmax,glmin,glmax,llmin,llmax,
            rlmax, TT, OCTG/(profile[:glmax+1]), 
            OCTL/(profile[:glmax+1]),
            gtype=gtype)[0]*profile)
        return 1./inv_r
    
    elif est=="Ksrc":
        inv_r = (norm_general.qtt_asym(
            "src",lmax,glmin,glmax,llmin,llmax,
            rlmax, TT, OCTG/(profile[:glmax+1]),
            OCTL/(profile[:glmax+1]),
            gtype=gtype)[0]*profile)
        return 1./inv_r

    else:
        return norm_general.xtt_asym(est,lmax,glmin,glmax,llmin,llmax,rlmax,
                                     TT, OCTG, OCTL, gtype=gtype)

def setup_asym_recon(px, lmin, lmax, mlmax,
                      cl_rksz, tcls_X, tcls_Y,
                      tcls_XY, do_lh=False,
                      do_psh=False):

    outputs = {}
    #CMB theory for filters
    ucls,_ = futils.get_theory_dicts(grad=True,
                                    lmax=mlmax)
    outputs["ucls"] = ucls
    #Get qfunc and normalization
    #profile is Cl**0.5
    profile = cl_rksz**0.5
    outputs["profile"] = profile

    def filter_X(X):
        return filter_T(X, cltot_X, lmin, lmax)
    outputs["filter_X"] = filter_X
    def filter_Y(Y):
        return filter_T(Y, cltot_Y, lmin, lmax)
    outputs["filter_Y"] = filter_Y

    print(tcls_X, tcls_Y, tcls_XY)
    cltot_X, cltot_Y, cltot_XY = (
        tcls_X[:lmax+1], tcls_Y[:lmax+1], tcls_XY[:lmax+1]
        )

    norm_args_XY = (mlmax, lmin, lmax, lmin,
        lmax, lmax, ucls['TT'][:lmax+1], cltot_Y,
        cltot_X)
    norm_phi_XY = norm_general.qtt_asym("lens", *norm_args_XY)

    outputs["norm_phi_XY"] = norm_phi_XY
    norm_K_XY = norm_qtt_asym(
        "src", *norm_args_XY, profile=profile)[0] #I think this should be the same as YX?
    outputs["norm_K_XY"] = norm_K_XY

    print("getting qe N0s")
    #Now get the N0s (need these for constructing
    #symmetrized estimator)
    #Note these are the noise on the *unnormalized*
    #estimators
    """
    wL_X = 1./cltot_X
    wL_Y = 1./cltot_Y
    wGK_X = (1./cltot_X[:lmax+1])/2
    wGK_Y = (1./cltot_Y[:lmax+1])/2
    """
    wLK_X = profile[:lmax+1]/cltot_X
    wLK_Y = profile[:lmax+1]/cltot_Y
    wGK_X = profile[:lmax+1]/cltot_X[:lmax+1]/2
    wGK_Y = profile[:lmax+1]/cltot_Y[:lmax+1]/2

    N0_XYXY_K_nonorm = noise_spec.qtt_asym(
        'src', mlmax, lmin, lmax,
         wLK_X, wGK_Y, wLK_X, wGK_Y,
         cltot_X,cltot_Y,cltot_XY,cltot_XY)[0]/profile**2
    
    #Normalize the N0
    N0_XYXY_K = N0_XYXY_K_nonorm * norm_K_XY**2

    outputs["N0_XYXY_K"] = N0_XYXY_K

    print("getting qe fg_trispectrum functions")
    #Also will be useful to define here functions to get the
    #tripsectrum N0 for foregrounds. 
    def get_fg_trispectrum_N0_XYXY(clfg_X, clfg_Y, clfg_XY):
        N0_tri_XYXY_prof_nonorm = noise_spec.qtt_asym(
            "src", mlmax,lmin,lmax,
            wLK_X, wGK_Y, wLK_X, wGK_Y,
            clfg_X[:lmax+1], clfg_Y[:lmax+1],
            clfg_XY[:lmax+1], clfg_XY[:lmax+1])[0]
        print(N0_tri_XYXY_prof_nonorm.shape)
        N0_tri_XYXY_prof_nonorm /= profile**2
        N0_tri_XYXY_prof = (
            N0_tri_XYXY_prof_nonorm
            *norm_K_XY*norm_K_XY)

        return N0_tri_XYXY_prof
    outputs["get_fg_trispectrum_N0_XYXY"] = get_fg_trispectrum_N0_XYXY

    #Ok, so we have norms and N0s
    #Now the qfuncs
    print("getting XY and YX qfuncs for qe")
    def get_XY_filtered(X_filtered, Y_filtered, 
                        X_nofilter=None, Y_nofilter=None):
        if X_filtered is None:
            X_filtered = filter_X(X_nofilter)
            Y_filtered = filter_Y(Y_nofilter)
        else:
            assert X_filtered is not None
            assert Y_filtered is not None
            assert Y_nofilter is None
        return X_filtered, Y_filtered
        #Now we can define the qfuncs                                                                                                                      
    def qfunc_K_XY(X_filtered, Y_filtered):
        K_nonorm = qe.qe_source(
            px, mlmax, X_filtered,
            xfTalm=Y_filtered, profile=profile)
        #and normalize                                                                                                                                 
        return curvedsky.almxfl(K_nonorm, norm_K_XY)
    outputs["qfunc_K_XY"] = qfunc_K_XY
    outputs["qfunc_K_XY_incfilter"] = lambda X,Y: qfunc_K_XY(filter_X(X), filter_Y(Y))
    
    return outputs


def get_N0_matrix_bh(
        N0_K, N0_K_Y, 
        N0_Y_K, N0_Y, 
        R_AB_inv, R_CD_inv):
    #these input N0s should be normalized!!!
    #and N0_Y should be for grad or curl only
    #i.e. not a tuple with both
    mlmax=len(N0_K)-1
    N0_matrix = np.zeros((len(N0_K), 2, 2))
    for N0 in (N0_K, N0_K_Y, N0_Y_K, N0_Y):
        assert N0.shape == N0_K.shape
    N0_matrix[:,0,0] = N0_K.copy()
    N0_matrix[:,0,1] = N0_K_Y.copy()
    N0_matrix[:,1,0] = N0_Y_K.copy()
    N0_matrix[:,1,1] = N0_Y.copy()

    #now the bh version
    N0_matrix_bh = np.zeros_like(N0_matrix)
    for l in range(mlmax+1):
        N0_matrix_bh[l] = np.dot(
            np.dot(R_AB_inv[l], N0_matrix[l]), (R_CD_inv[l]).T)
    #0,0 element is the Y_bh N0
    return N0_matrix_bh



def setup_ABCD_recon(px, lmin, lmax, mlmax,
                      cl_rksz, cltot_A, cltot_B,
                      cltot_C, cltot_D,
                      cltot_AC, cltot_BD,
                      cltot_AD, cltot_BC, do_lh=False,
                      do_psh=False, divide_by_2uL=False):
    print("############")
    print("do_psh:",do_psh)
    print("#############")
    outputs = {}
    #CMB theory for filters
    ucls,_ = futils.get_theory_dicts(grad=True,
                                    lmax=mlmax)
    outputs["ucls"] = ucls
    outputs["cl_rksz"] = cl_rksz
    #Get qfunc and normalization
    #profile is Cl**0.5
    profile = cl_rksz**0.5
    profile[:lmin] = profile[lmin] #funny things can happen with noisy Cls at low l
    outputs["profile"] = profile

    outputs["lmin"] = lmin
    outputs["lmax"] = lmax
    outputs["mlmax"] = mlmax
    
    #also save all the total cls
    outputs["cltot_A"] = cltot_A
    outputs["cltot_B"] = cltot_B
    outputs["cltot_C"] = cltot_C
    outputs["cltot_D"] = cltot_D
    outputs["cltot_AC"] = cltot_AC
    outputs["cltot_BD"] = cltot_BD
    outputs["cltot_AD"] = cltot_AD
    outputs["cltot_BC"] = cltot_BC
    
    def filter_A(X):
        return filter_T(X, cltot_A, lmin, lmax)
    outputs["filter_A"] = filter_A
    def filter_B(X):
        return filter_T(X, cltot_B, lmin, lmax)
    outputs["filter_B"] = filter_B
    def filter_C(X):
        return filter_T(X, cltot_C, lmin, lmax)
    outputs["filter_C"] = filter_C
    def filter_D(X):
        return filter_T(X, cltot_D, lmin, lmax)
    outputs["filter_D"] = filter_D

    #cltot_X, cltot_Y, cltot_XY = (
    #    tcls_X[:lmax+1], tcls_Y[:lmax+1], tcls_XY[:lmax+1]
    #    )

    norm_args_AB = (mlmax, lmin, lmax, lmin,
        lmax, lmax, ucls['TT'][:lmax+1], cltot_B[:lmax+1],
        cltot_A[:lmax+1])

    norm_K_AB = norm_qtt_asym(
        "src", *norm_args_AB, profile=profile)[0]
    norm_phi_AB = norm_qtt_asym(
        "lens", *norm_args_AB)
    norm_src_AB = norm_qtt_asym(
        "src", *norm_args_AB)[0]

    outputs["norm_K_AB"] = norm_K_AB
    outputs["norm_phi_AB"] = norm_phi_AB #note this has grad and curl, keep both for now
    outputs["norm_src_AB"] = norm_src_AB
    
    norm_args_CD = (mlmax, lmin, lmax, lmin,
        lmax, lmax, ucls['TT'][:lmax+1], cltot_D[:lmax+1],
        cltot_C[:lmax+1])

    norm_K_CD = norm_qtt_asym(
        "src", *norm_args_CD, profile=profile)[0] #I think this should be the same as YX?
    norm_phi_CD = norm_qtt_asym(
        "lens", *norm_args_CD)
    norm_src_CD = norm_qtt_asym(
        "src", *norm_args_CD)[0]
    outputs["norm_K_CD"] = norm_K_CD
    outputs["norm_phi_CD"] = norm_phi_CD    
    outputs["norm_src_CD"] = norm_src_CD
    
    print("getting qe N0s")
    #Now get the N0s (need these for constructing
    #symmetrized estimator)
    #Note these are the noise on the *unnormalized*
    #estimators
    """
    wL_X = 1./cltot_X
    wL_Y = 1./cltot_Y
    wGK_X = (1./cltot_X[:lmax+1])/2
    wGK_Y = (1./cltot_Y[:lmax+1])/2
    """
    wLK_A = profile[:lmax+1]/cltot_A[:lmax+1]
    wLK_C = profile[:lmax+1]/cltot_C[:lmax+1]
    wGK_D = profile[:lmax+1]/cltot_D[:lmax+1]/2
    wGK_B = profile[:lmax+1]/cltot_B[:lmax+1]/2

    N0_ABCD_K_nonorm = noise_spec.qtt_asym(
        'src', mlmax, lmin, lmax,
         wLK_A, wGK_B, wLK_C, wGK_D,
         cltot_AC[:lmax+1], cltot_BD[:lmax+1], cltot_AD[:lmax+1], cltot_BC[:lmax+1])[0]/profile**2
    #Normalize the N0
    N0_ABCD_K = N0_ABCD_K_nonorm * norm_K_AB * norm_K_CD
    outputs["N0_ABCD_K"] = N0_ABCD_K

    if do_lh:
        wLphi_A = 1./cltot_A[:lmax+1]
        wLphi_C = 1./cltot_C[:lmax+1]
        wGphi_D = (ucls['TT'][:lmax+1]/cltot_D[:lmax+1])
        wGphi_B = (ucls['TT'][:lmax+1]/cltot_B[:lmax+1])
        #get N0s, responses etc.
        N0_ABCD_phi_nonorm = noise_spec.qtt_asym(
            'lens', mlmax, lmin, lmax,
             wLphi_A, wGphi_B, wLphi_C, wGphi_D,
             cltot_AC[:lmax+1], cltot_BD[:lmax+1], cltot_AD[:lmax+1], cltot_BC[:lmax+1])
        #Normalize the N0
        N0_ABCD_phi = (N0_ABCD_phi_nonorm[0] * norm_phi_AB[0] * norm_phi_CD[0],
                       N0_ABCD_phi_nonorm[1] * norm_phi_AB[1] * norm_phi_CD[1])
        outputs["N0_ABCD_phi"] = N0_ABCD_phi
        #now the responses
        R_K_phi_AB = norm_xtt_asym(
            "srclens", *norm_args_AB, profile=profile)
        outputs["R_K_phi_AB"] = R_K_phi_AB
        
        R_phi_K_AB = norm_xtt_asym(
            "lenssrc", *norm_args_AB, profile=profile)
        outputs["R_phi_K_AB"] = R_phi_K_AB
        
        R_K_phi_CD = norm_xtt_asym(
            "srclens", *norm_args_CD, profile=profile)
        outputs["R_K_phi_CD"] = R_K_phi_CD
        
        R_phi_K_CD = norm_xtt_asym(
            "lenssrc", *norm_args_CD, profile=profile)
        outputs["R_phi_K_CD"] = R_phi_K_CD


    if do_psh:
        wLs_A = 1./cltot_A[:lmax+1]
        wLs_C = 1./cltot_C[:lmax+1]
        wGs_D = 1./cltot_D[:lmax+1]/2
        wGs_B = 1./cltot_B[:lmax+1]/2
        N0_ABCD_src_nonorm = noise_spec.qtt_asym(
            'src', mlmax, lmin, lmax,
            wLs_A, wGs_B, wLs_C, wGs_D,
            cltot_AC[:lmax+1], cltot_BD[:lmax+1], cltot_AD[:lmax+1], cltot_BC[:lmax+1])[0]
        N0_ABCD_src = N0_ABCD_src_nonorm * norm_src_AB * norm_src_CD
        outputs["N0_ABCD_src"] = N0_ABCD_src
        #now the responses
        R_K_src_AB = norm_xtt_asym(
            "Ksrc", *norm_args_AB, profile=profile)
        outputs["R_K_src_AB"] = R_K_src_AB
        R_src_K_AB = norm_xtt_asym(
            "srcK", *norm_args_AB, profile=profile)
        outputs["R_src_K_AB"] = R_src_K_AB
        
        R_K_src_CD = norm_xtt_asym(
            "Ksrc", *norm_args_CD, profile=profile)
        outputs["R_K_src_CD"] = R_K_src_CD
        R_src_K_CD = norm_xtt_asym(
            "srcK", *norm_args_CD, profile=profile)
        outputs["R_src_K_CD"] = R_src_K_CD
        
    
    print("getting qe fg_trispectrum functions")
    #Also will be useful to define here functions to get the
    #tripsectrum N0 for foregrounds. 
    def get_fg_trispectrum_N0_ABCD(clfg_AC, clfg_BD, clfg_AD, clfg_BC):
        N0_tri_ABCD_prof_nonorm = noise_spec.qtt_asym(
            "src", mlmax,lmin,lmax,
            wLK_A, wGK_B, wLK_C, wGK_D,
            clfg_AC[:lmax+1], clfg_BD[:lmax+1],
            clfg_AD[:lmax+1], clfg_BC[:lmax+1])[0]
        print(N0_tri_ABCD_prof_nonorm.shape)
        N0_tri_ABCD_prof_nonorm /= profile**2
        N0_tri_ABCD_prof = (
            N0_tri_ABCD_prof_nonorm
            *norm_K_AB*norm_K_CD)

        return N0_tri_ABCD_prof
    outputs["get_fg_trispectrum_N0_ABCD"] = get_fg_trispectrum_N0_ABCD

    #Ok, so we have norms and N0s
    #Now the qfuncs
    print("getting AB and CD qfuncs for qe")
    def qfunc_K_AB(A_filtered, B_filtered):
        K_nonorm = qe.qe_source(
            px, mlmax, A_filtered,
            xfTalm=B_filtered, profile=profile)
        #and normalize
        if divide_by_2uL:
            raise ValueError("not yet implemeneted divide_by_2uL")
            norm = norm_K_AB / 2 / profile
        else:
            norm = norm_K_AB
        return curvedsky.almxfl(K_nonorm, norm)

    def qfunc_K_CD(C_filtered, D_filtered):
        K_nonorm = qe.qe_source(
            px, mlmax, C_filtered,
            xfTalm=D_filtered, profile=profile)
        #and normalize                                                                                                             
        if divide_by_2uL:
            norm = norm_K_CD / 2 / profile
        else:
            norm = norm_K_CD
        return curvedsky.almxfl(K_nonorm, norm_K_CD)

    outputs["qfunc_K_AB"] = qfunc_K_AB
    outputs["qfunc_K_AB_incfilter"] = lambda X,Y: qfunc_K_AB(filter_A(X), filter_B(Y))
    outputs["qfunc_K_CD"] = qfunc_K_CD
    outputs["qfunc_K_CD_incfilter"] = lambda X,Y: qfunc_K_CD(filter_C(X), filter_D(Y))

    def get_inverse_response_matrix(norm_K, norm_phi, R_K_phi, R_phi_K):
        R = np.ones((mlmax+1, 2, 2))
        R[:,0,1] = (norm_K * R_K_phi).copy()
        R[:,1,0] = (norm_phi * R_phi_K).copy()
        R_inv = np.zeros_like(R)
        for l in range(mlmax+1):
            R_inv[l] = np.linalg.inv(R[l])
        return R_inv
    
    if do_lh:

        
        def get_qfunc_K_XY_lh(qfunc_K_XY, norm_K_XY, norm_phi_XY,
                              R_matrix_XY_inv):
            
            def qfunc_phi_XY(X_filtered,Y_filtered):
                phi_nonorm = qe.qe_all(px,ucls,mlmax,
                                    fTalm=X_filtered,fEalm=None,fBalm=None,
                                    estimators=['TT'],
                                    xfTalm=Y_filtered,xfEalm=None,xfBalm=None)['TT']
                return (curvedsky.almxfl(phi_nonorm[0], norm_phi_XY[0]),
                        curvedsky.almxfl(phi_nonorm[1], norm_phi_XY[1]))

            def qfunc_K_XY_lh(X_filtered, Y_filtered):

                phi_XY = qfunc_phi_XY(X_filtered, Y_filtered)
                K_XY_nobh = qfunc_K_XY(X_filtered, Y_filtered)
                K_XY_bh = (curvedsky.almxfl(K_XY_nobh, R_matrix_XY_inv[:,0,0])
                          +curvedsky.almxfl(phi_XY[0], R_matrix_XY_inv[:,0,1])
                          )
                return K_XY_bh

            return qfunc_K_XY_lh, qfunc_phi_XY

        R_matrix_AB_inv_phi = get_inverse_response_matrix(
                norm_K_AB, norm_phi_AB[0], R_K_phi_AB, R_phi_K_AB)
        R_matrix_CD_inv_phi = get_inverse_response_matrix(
                norm_K_CD, norm_phi_CD[0], R_K_phi_CD, R_phi_K_CD)       
        
        qfunc_K_AB_lh, qfunc_phi_AB = get_qfunc_K_XY_lh(qfunc_K_AB, norm_K_AB, norm_phi_AB,
                                          R_matrix_AB_inv_phi)
        qfunc_K_CD_lh, qfunc_phi_CD = get_qfunc_K_XY_lh(qfunc_K_CD, norm_K_CD, norm_phi_CD,
                                          R_matrix_CD_inv_phi)
        outputs["qfunc_K_AB_lh"] = qfunc_K_AB_lh
        outputs["qfunc_K_CD_lh"] = qfunc_K_CD_lh
        outputs["qfunc_phi_AB"] = qfunc_phi_AB
        outputs["qfunc_phi_CD"] = qfunc_phi_CD
        
        N0_ABCD_K_phi_nonorm = noise_spec.xtt_asym(
            "srclens", mlmax,lmin,lmax,
            wLK_A, wGK_B, wLphi_C, wGphi_D,
            cltot_AC[:lmax+1], cltot_BD[:lmax+1], cltot_AD[:lmax+1], cltot_BC[:lmax+1])/profile
        N0_ABCD_K_phi = (
            N0_ABCD_K_phi_nonorm
            *norm_K_AB*norm_phi_CD[0]*1)  #factor 2 because apparently don't need 1/2 in wGs
        
        N0_ABCD_phi_K_nonorm = noise_spec.xtt_asym(
            "lenssrc", mlmax,lmin,lmax,
            wLphi_A, wGphi_B, wLK_C, wGK_D,
            cltot_AC[:lmax+1], cltot_BD[:lmax+1], cltot_AD[:lmax+1], cltot_BC[:lmax+1])/profile
        N0_ABCD_phi_K = (
            N0_ABCD_phi_K_nonorm
            *norm_phi_AB[0]*norm_K_CD*1)  #factor 2 because apparently don't need 1/2 in wGs

        N0_ABCD_K_lh = get_N0_matrix_bh(
            N0_ABCD_K, N0_ABCD_K_phi, N0_ABCD_phi_K, N0_ABCD_phi[0],
            R_matrix_AB_inv_phi, R_matrix_CD_inv_phi)[:,0,0]
        outputs["N0_ABCD_K_lh"] = N0_ABCD_K_lh

        def get_fg_trispectrum_N0_ABCD_lh(clfg_AC, clfg_BD, clfg_AD, clfg_BC):

            N0_tri_ABCD_K = get_fg_trispectrum_N0_ABCD(
                clfg_AC, clfg_BD, clfg_AD, clfg_BC)
            
            N0_tri_ABCD_phi_nonorm = noise_spec.qtt_asym(
                'lens', mlmax, lmin, lmax,
                wLphi_A, wGphi_B, wLphi_C, wGphi_D,
                clfg_AC[:lmax+1], clfg_BD[:lmax+1],
                clfg_AD[:lmax+1], clfg_BC[:lmax+1])
            N0_tri_ABCD_phi = (N0_tri_ABCD_phi_nonorm[0] * norm_phi_AB[0] * norm_phi_CD[0],
                               N0_tri_ABCD_phi_nonorm[1] * norm_phi_AB[1] * norm_phi_CD[1])
            
            N0_tri_ABCD_K_phi_nonorm = noise_spec.xtt_asym(
                "srclens", mlmax,lmin,lmax,
                wLK_A, wGK_B, wLphi_C, wGphi_D,
                clfg_AC[:lmax+1], clfg_BD[:lmax+1], clfg_AD[:lmax+1], clfg_BC[:lmax+1])/profile
            N0_tri_ABCD_K_phi = (
                N0_tri_ABCD_K_phi_nonorm * norm_K_AB * norm_phi_CD[0]*1 #factor 2 because apparently don't need 1/2 in wGs 
                )
            
            N0_tri_ABCD_phi_K_nonorm = noise_spec.xtt_asym(
                "lenssrc", mlmax,lmin,lmax,
                wLphi_A, wGphi_B, wLK_C, wGK_D,
                clfg_AC[:lmax+1], clfg_BD[:lmax+1], clfg_AD[:lmax+1], clfg_BC[:lmax+1])/profile
            N0_tri_ABCD_phi_K = (
                N0_tri_ABCD_phi_K_nonorm
                *norm_phi_AB[0]*norm_K_CD*1)	#factor 2 because apparently don't need 1/2 in wGs


            #N0_tri_ABCD_phi_nonorm = noise_spec.qtt_asym(
            #    'lens', mlmax, lmin, lmax,
            #     wLphi_A, wGphi_B, wLphi_C, wGphi_D,
            #     clfg_AC[:lmax+1], clfg_BD[:lmax+1], clfg_AD[:lmax+1], clfg_BC[:lmax+1])
            ##Normalize the N0                                                                                                                                                                                                
            #N0_tri_ABCD_phi = (N0_tri_ABCD_phi_nonorm[0] * norm_phi_AB[0] * norm_phi_CD[0],
            #               N0_tri_ABCD_phi_nonorm[1] * norm_phi_AB[1] * norm_phi_CD[1])
            
            N0_tri_ABCD_K_lh = get_N0_matrix_bh(
                N0_tri_ABCD_K, N0_tri_ABCD_K_phi, N0_tri_ABCD_phi_K, N0_tri_ABCD_phi[0],
                R_matrix_AB_inv_phi, R_matrix_CD_inv_phi)[:,0,0]
            return N0_tri_ABCD_K_lh
        
        outputs["get_fg_trispectrum_N0_ABCD_lh"] = get_fg_trispectrum_N0_ABCD_lh
        
    if do_psh:

        """
        def get_N0_matrix_psh(
                N0_K, N0_K_src, 
                N0_src_K, N0_src, 
                R_AB_inv, R_CD_inv):
            #these input N0s should be normalized!!!
            #and N0_src should be for grad or curl only
            #i.e. not a tuple with both

            N0_matrix = np.zeros((mlmax+1, 2, 2))
            for N0 in (N0_K, N0_K_src, N0_src_K, N0_src):
                assert N0.shape == (mlmax+1,)
            N0_matrix[:,0,0] = N0_K.copy()
            N0_matrix[:,0,1] = N0_K_src.copy()
            N0_matrix[:,1,0] = N0_src_K.copy()
            N0_matrix[:,1,1] = N0_src.copy()

            #now the lh version
            N0_matrix_psh = np.zeros_like(N0_matrix)
            for l in range(mlmax+1):
                N0_matrix_psh[l] = np.dot(
                    np.dot(R_AB_inv[l], N0_matrix[l]), (R_CD_inv[l]).T)
            #0,0 element is the phi_bh N0
            return N0_matrix_psh
        """
        
        def get_qfunc_K_XY_psh(qfunc_K_XY, norm_K_XY, norm_src_XY,
                              R_matrix_XY_inv):
            
            def qfunc_src_XY(X_filtered,Y_filtered):
                src_nonorm = qe.qe_source(px, mlmax,
                                          fTalm=X_filtered, xfTalm=Y_filtered)
                return curvedsky.almxfl(src_nonorm, norm_src_XY)

            def qfunc_K_XY_psh(X_filtered, Y_filtered):

                src_XY = qfunc_src_XY(X_filtered, Y_filtered)
                K_XY_nobh = qfunc_K_XY(X_filtered, Y_filtered)
                K_XY_bh = (curvedsky.almxfl(K_XY_nobh, R_matrix_XY_inv[:,0,0])
                          +curvedsky.almxfl(src_XY, R_matrix_XY_inv[:,0,1])
                          )
                return K_XY_bh

            return qfunc_K_XY_psh, qfunc_src_XY

        R_matrix_AB_inv_src = get_inverse_response_matrix(
                norm_K_AB, norm_src_AB, R_K_src_AB, R_src_K_AB)
        R_matrix_CD_inv_src = get_inverse_response_matrix(
                norm_K_CD, norm_src_CD, R_K_src_CD, R_src_K_CD)
        #print("R_matrix_AB_inv_src",R_matrix_AB_inv_src)
        
        qfunc_K_AB_psh, qfunc_src_AB  = get_qfunc_K_XY_psh(qfunc_K_AB, norm_K_AB, norm_src_AB,
                                          R_matrix_AB_inv_src)
        qfunc_K_CD_psh, qfunc_src_CD = get_qfunc_K_XY_psh(qfunc_K_CD, norm_K_CD, norm_src_CD,
                                          R_matrix_CD_inv_src)
        outputs["qfunc_K_AB_psh"] = qfunc_K_AB_psh
        outputs["qfunc_K_CD_psh"] = qfunc_K_CD_psh
        outputs["qfunc_src_AB"] = qfunc_src_AB
        outputs["qfunc_src_CD"] = qfunc_src_CD
        
        N0_ABCD_K_src_nonorm = noise_spec.qtt_asym(
            "src", mlmax,lmin,lmax,
            wLK_A, wGK_B, wLs_C, wGs_D,
            cltot_AC[:lmax+1], cltot_BD[:lmax+1],
            cltot_AD[:lmax+1], cltot_BC[:lmax+1])[0]/profile
        N0_ABCD_K_src = (
            N0_ABCD_K_src_nonorm
            *norm_K_AB*norm_src_CD)
        
        outputs["N0_ABCD_K_src"] = N0_ABCD_K_src
        
        N0_ABCD_src_K_nonorm = noise_spec.qtt_asym(
            "src", mlmax,lmin,lmax,
            wLs_A, wGs_B, wLK_C, wGK_D,
            cltot_AC[:lmax+1], cltot_BD[:lmax+1], cltot_AD[:lmax+1], cltot_BC[:lmax+1])[0]/profile
        N0_ABCD_src_K = (
            N0_ABCD_src_K_nonorm
            * norm_src_AB * norm_K_CD)
        
        outputs["N0_ABCD_src_K_nonorm"] = N0_ABCD_src_K_nonorm

        N0_ABCD_K_psh = get_N0_matrix_bh(
            N0_ABCD_K, N0_ABCD_K_src, N0_ABCD_src_K, N0_ABCD_src,
            R_matrix_AB_inv_src, R_matrix_CD_inv_src)[:,0,0]
        outputs["N0_ABCD_K_psh"] = N0_ABCD_K_psh
            
        def get_fg_trispectrum_N0_ABCD_psh(clfg_AC, clfg_BD, clfg_AD, clfg_BC):

            N0_tri_ABCD_K = get_fg_trispectrum_N0_ABCD(
                clfg_AC, clfg_BD, clfg_AD, clfg_BC)
            
            N0_tri_ABCD_src_nonorm = noise_spec.qtt_asym(
                'src', mlmax, lmin, lmax,
                wLs_A, wGs_B, wLs_C, wGs_D,
                clfg_AC[:lmax+1], clfg_BD[:lmax+1], clfg_AD[:lmax+1], clfg_BC[:lmax+1])[0]
            N0_tri_ABCD_src = N0_tri_ABCD_src_nonorm * norm_src_AB * norm_src_CD
            
            N0_tri_ABCD_K_src_nonorm = noise_spec.qtt_asym(
                "src", mlmax,lmin,lmax,
                wLK_A, wGK_B, wLs_C, wGs_D,
                clfg_AC[:lmax+1], clfg_BD[:lmax+1], clfg_AD[:lmax+1], clfg_BC[:lmax+1])[0]/profile
            N0_tri_ABCD_K_src = (
                N0_tri_ABCD_K_src_nonorm
                *norm_K_AB*norm_src_CD)
            
            N0_tri_ABCD_src_K_nonorm = noise_spec.qtt_asym(
                "src", mlmax,lmin,lmax,
                wLs_A, wGs_B, wLK_C, wGK_D,
                clfg_AC[:lmax+1], clfg_BD[:lmax+1], clfg_AD[:lmax+1], clfg_BC[:lmax+1])[0]/profile
            N0_tri_ABCD_src_K = (
                N0_tri_ABCD_src_K_nonorm
                * norm_src_AB * norm_K_CD)


            N0_tri_ABCD_K_psh = get_N0_matrix_bh(
                N0_tri_ABCD_K, N0_tri_ABCD_K_src, N0_tri_ABCD_src_K, N0_tri_ABCD_src,
                R_matrix_AB_inv_src, R_matrix_CD_inv_src)[:,0,0]

            return N0_tri_ABCD_K_psh
        
        outputs["get_fg_trispectrum_N0_ABCD_psh"] = get_fg_trispectrum_N0_ABCD_psh
            
    return outputs
    

def setup_recon(px, lmin, lmax, mlmax,
                cl_rksz, tcls_X, tcls_Y=None,
                tcls_XY=None, do_lh=False,
                do_psh=False):

    outputs = {}
    
    #CMB theory for filters
    ucls,_ = futils.get_theory_dicts(grad=True,
                                    lmax=mlmax)
    outputs["ucls"] = ucls
    #Get qfunc and normalization
    #profile is Cl**0.5
    profile = cl_rksz**0.5
    outputs["profile"] = profile

    def filter_X(X):
        return filter_T(X, tcls_X["TT"], lmin, lmax)
    outputs["filter_X"] = filter_X
    """
    def filter_alms_X(alms):
        if len(alms)!=3:
            alms = dummy_teb(alms)
        alms_filtered = futils.isotropic_filter(alms,
                tcls_X, lmin, lmax, ignore_te=True)
        return alms_filtered
    outputs["filter_alms_X"] = filter_alms_X
    """
    
    if tcls_Y is None:
        norm_K = pytempura.get_norms(
            ['src'], ucls, tcls_X,
            lmin, lmax, k_ellmax=mlmax,
            profile=profile)['src']
        norm_K[0]=0.
        norm_qtt_asym
        outputs["norm_K"] = norm_K

        #The Smith and Ferraro estimator Ksf differs
        #slightly from our profile estimator,K
        #Ksf = 2*profile*K (both unnormalised)
        #So normalizations should be related via:
        norm_Ksf = norm_K / 2 / profile
        outputs["norm_Ksf"] = norm_Ksf

        #For the normalized estimator this
        #is also the N0.
        N0_K = norm_K / profile**2
        outputs["N0_K"] = N0_K 
        
        #otherwise, it is
        N0_K_nonorm = 1./norm_K

        print('getting point-source norm')
        print(np.any(tcls_X['TT']==0.))
        print(np.any(~np.isfinite(tcls_X['TT'])))
        norm_ps = pytempura.get_norms(
            ['src'], ucls, tcls_X,
            lmin, lmax, k_ellmax=mlmax)['src']
        outputs["norm_ps"] = norm_ps

        #Also get lensing norms and cross-response for
        #lensing hardening
        print('getting lensing norm')
        norm_phi = pytempura.norm_lens.qtt(
            mlmax, lmin, lmax, ucls['TT'],
            tcls_X['TT'], gtype='')[0]
        outputs["norm_phi"] = norm_phi

        print('getting source-lens response')
        print(tcls_X['TT'])
        print(profile)
        R_K_phi = pytempura.get_cross(
            'SRC','TT', ucls, tcls_X,
            lmin, lmax, k_ellmax=mlmax,
            profile=profile)
        outputs["R_K_phi"] = R_K_phi

        print('getting source-point-source response')
        R_K_ps = (1./pytempura.get_norms(
            ['src'], ucls, tcls_X,
            lmin, lmax, k_ellmax=mlmax,
            profile = profile**0.5)['src'])
        R_K_ps[0] = 0.
        outputs["R_K_ps"] = R_K_ps

        #unnormalized source estimator
        def qfunc_prof(X, Y):
            return qe.qe_source(px,mlmax,profile=profile,
                             fTalm=Y,xfTalm=X)

        #Smith/Ferraro estimator (x 2!)
        def qfunc_Ksf_nonorm(X,Y):
            s_nonorm = qfunc_prof(X, Y)
            K_nonorm = curvedsky.almxfl(s_nonorm, norm_Ksf)
            return K_nonorm

        def qfunc_K(X,Y):
            K_nonorm = qfunc_prof(X,Y)
            return curvedsky.almxfl(
                K_nonorm, norm_K
                )
        outputs["qfunc_K"] = qfunc_K
        qfunc_K_incfilter = lambda X,Y: qfunc_K(filter_alms_X(X)[0],
                                                filter_alms_X(Y)[0])
        outputs["qfunc_K_incfilter"] = qfunc_K_incfilter

        def get_fg_trispectrum_K_N0(cl_fg):
            #N0 is (A^K)^2 / A_fg (see eqn. 9 of 1310.7023
            #for the normal estimator, and a bit more complex for
            #the bias hardened case
            Ctot = tcls_X['TT']**2 / cl_fg
            norm_fg = pytempura.get_norms(
                ['src'], ucls, {"TT" : Ctot},
                lmin, lmax, k_ellmax=mlmax,
                profile=profile)['src']
            
            return N0_K**2 / norm_fg
        outputs["get_fg_trispectrum_K_N0"] = get_fg_trispectrum_K_N0

        #qfunc for lensing
        qfunc_phi_nonorm = lambda X,Y: qe.qe_all(
            px,ucls,mlmax,fTalm=Y,fEalm=None,fBalm=None,
            estimators=['TT'], xfTalm=X, xfEalm=None,
            xfBalm=None)['TT']

        def qfunc_phi(X,Y):
            phi = qfunc_phi_nonorm(X,Y)
            return curvedsky.almxfl(phi[0], norm_phi)
        outputs["qfunc_phi"] = qfunc_phi
        
        def qfunc_lh(X,Y):
            #(K_est  )  = ( 1       A^K R^K\phi )(K   )
            #(\phi_est)    ( A^\phi R^K\phi    1 )(\phi)
            # So bias-hardened K_BH given by
            # K_BH = (K_est - A^K R^K\phi \phi_est) / (1 - A^K*A^\phi*(R^K\phi)^2)
            K = qfunc_K(X,Y)
            phi = qfunc_phi(X,Y)
            num = K - curvedsky.almxfl(
                phi, norm_K * R_K_phi)
            denom = 1 - norm_K * norm_phi * R_K_phi**2
            K_normed_lh = curvedsky.almxfl(num, 1./denom)
            return K_normed_lh

        outputs["qfunc_K_lh"] = qfunc_lh
        qfunc_lh_incfilter = lambda X,Y: qfunc_lh(filter_alms_X(X),
                                                  filter_alms_X(Y))
        outputs["qfunc_lh_incfilter"] = qfunc_lh_incfilter

        
        N0_K_lh = N0_K / (1 - norm_K * norm_phi * R_K_phi**2)
        outputs["N0_K_lh"] = N0_K_lh
        def get_fg_trispectrum_K_N0_lh(cl_fg):
            Ctot = tcls_X['TT']**2 / cl_fg
            norm_fg = pytempura.get_norms(
                ['src'], ucls, {"TT" : Ctot},
                lmin, lmax, k_ellmax=mlmax,
                profile=profile)['src']
            norm_phi_fg = pytempura.get_norms(
                ['TT'], ucls, {'TT':Ctot},
                lmin, lmax,
                k_ellmax=mlmax)['TT'][0]
            R_K_phi_fg = pytempura.get_cross(
                'SRC','TT', ucls, {'TT':Ctot},
                lmin, lmax,
                k_ellmax=mlmax, profile=profile)
            N0 = N0_K**2/norm_fg
            N0_phi = norm_phi**2/norm_phi_fg
            N0_tri = (N0 + R_K_phi**2 * norm_K**2 * N0_phi
                      - 2 * R_K_phi * norm_K**2 * norm_phi * R_K_phi_fg)
            N0_tri /= (1 - norm_K*norm_phi*R_K_phi**2)**2 
            return N0_tri
        outputs["get_fg_trispectrum_K_N0_lh"] = get_fg_trispectrum_K_N0_lh
        
        def qfunc_ps_nonorm(X,Y):
            return qe.qe_source(px, mlmax,
                                fTalm=Y,xfTalm=X)
        def qfunc_ps(X,Y):
            ps = qfunc_ps_nonorm(X, Y)
            return curvedsky.almxfl(ps, norm_ps)
        outputs["qfunc_ps"] = qfunc_ps

        def qfunc_psh(X,Y):
            #(K_est  )  = ( 1       A^K R^K\phi )(K   )
            #(\phi_est)    ( A^\phi R^K\phi    1 )(\phi)
            # So bias-hardened K_BH given by
            # K_BH = (K_est - A^K R^K\phi \phi_est) / (1 - A^K*A^\phi*(R^K\phi)^2)
            K = qfunc_K(X,Y)
            ps = qfunc_ps(X,Y)
            num = K - curvedsky.almxfl(ps, norm_K*R_K_ps)
            denom = 1 - norm_K * norm_ps * R_K_ps**2
            K_psh = curvedsky.almxfl(num, 1./denom)
            return K_psh
        
        outputs["qfunc_K_psh"] = qfunc_psh
        qfunc_psh_incfilter = lambda X,Y: qfunc_psh(filter_alms_X(X),
                                                  filter_alms_X(Y))
        outputs["qfunc_psh_incfilter"] = qfunc_psh_incfilter

        N0_K_psh = N0_K / (1 - norm_K * norm_ps * R_K_ps**2)
        outputs["N0_K_psh"] = N0_K_psh


        def get_fg_trispectrum_K_N0_psh(cl_fg):
            Ctot = tcls_X['TT']**2 / cl_fg
            norm_fg = pytempura.get_norms(
                ['src'], ucls, {"TT" : Ctot},
                lmin, lmax, k_ellmax=mlmax,
                profile=profile)['src']
            norm_ps_fg = pytempura.get_norms(
                ['src'], ucls, {'TT':Ctot},
                lmin, lmax,
                k_ellmax=mlmax)['src']
            print("norm_ps_fg.shape:",norm_ps_fg.shape)
            R_K_ps_fg = (1./pytempura.get_norms(
                ['src'], ucls, {'TT':Ctot},
                lmin, lmax, k_ellmax=mlmax,
                profile = profile**0.5)['src'])
            R_K_ps_fg[0] = 0.
            N0 = N0_K**2/norm_fg
            N0_ps = norm_ps**2/norm_ps_fg
            N0_tri = (N0 + R_K_ps**2 * norm_K**2 * N0_ps
                      - 2 * R_K_ps * norm_K**2 * norm_ps * R_K_ps_fg)
            N0_tri /= (1 - norm_K*norm_ps*R_K_ps**2)**2
            return N0_tri
        outputs["get_fg_trispectrum_K_N0_psh"] = get_fg_trispectrum_K_N0_psh

        return outputs

    
        assert prep_map_config['disable_noise']
        def get_sim_alm(seed, add_kszr=True,
                        add_cmb=True,
                        cl_fg_func=None,
                        lensed_cmb=True):
            print("calling get_sim_alm with")
            print("add_kszr:",add_kszr)
            print("add_cmb:",add_cmb)
            print("lensed_cmb:",lensed_cmb)
            if cl_fg_func is not None:
                print("adding Gaussian fgs")
            #generate alms
            #seed has form (icov, iset, inoise)
            #we just need one number - the following
            #should ensure we get unique realizations
            #for rdn0
            ksz2_seed = seed[1]*9999 + seed[2]
            if add_kszr:
                sim_alms = curvedsky.rand_alm(cl_smooth,
                                              seed=ksz2_seed)
            else:
                sim_alms = np.zeros(hp.Alm.getsize(mlmax),
                                    dtype=np.complex128)

            s_i,s_set,noise_seed = solenspipe.convert_seeds(seed)
            if add_cmb:
                if lensed_cmb:
                    cmb_alms = solenspipe.get_cmb_alm(s_i, s_set)[0]
                else:
                    cmb_alms = get_cmb_alm_unlensed(s_i, s_set)[0]
                cmb_alms = futils.change_alm_lmax(cmb_alms.astype(np.complex128), mlmax)
                sim_alms += cmb_alms

            if cl_fg_func is not None:
                print("adding Gaussian foreground power:")
                cl_fg = cl_fg_func(np.arange(mlmax+1))
                print(cl_fg)
                fg_alms = curvedsky.rand_alm(
                    cl_fg, seed=ksz2_seed)
                sim_alms += fg_alms

            #filter and return
            return filter_alms(sim_alms)

        outputs["get_sim_alm"] = get_sim_alm
        
        recon_stuff = {"shape" : shape,
                       "wcs" : wcs,
                       "norm_prof" : norm_prof,
                       "norm_K" : norm_K,
                       "ucls" : ucls,
                       "tcls" : tcls,
                       "qfunc_Ksf_nonorm" : qfunc_Ksf_nonorm,
                       "qfunc_K" : qfunc_K,
                       "qfunc_lh_sf" : qfunc_lh,
                       "qfunc_lh_normed" : qfunc_lh_normed,
                       "qfunc_psh_sf" : qfunc_psh,
                       "qfunc_psh_normed" : qfunc_psh_normed,
                       "qfunc_lens" : qfunc_lens,
                       "filter_alms" : filter_alms,
                       "get_sim_alm" : get_sim_alm,
                       "cl_total":cl_total,
                       "cl_kszr": cl_smooth,
                       "cl_total_cmb": cl_total_cmb,
                       "profile": profile,
                       "norm_ps": norm_ps,
                       "norm_lens" : norm_lens,
                       "R_prof_ps" : R_prof_ps,
                       "R_prof_tt" : R_prof_tt,
                       "N0_K_normed" : N0_K_normed,
                       "N0_K_nonorm" : N0_K_nonorm,
                       "N0_K_lh_normed" : N0_K_lh_normed,
                       "N0_K_lh_nonorm" : N0_K_lh_nonorm,
                       "N0_K_psh_normed" : N0_K_psh_normed,
                       "N0_K_psh_nonorm" : N0_K_psh_nonorm,
            }

        return recon_stuff
