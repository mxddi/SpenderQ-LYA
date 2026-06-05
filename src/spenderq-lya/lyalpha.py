'''

methods for identifying LyA absorption 


'''
import numpy as np 
from . import util as U


def identify_absorp(wobs, fobs, iobs, zobs, wmod, fmod, sigma_lya=1.5, sigma_lyb=1.5, wave_lya=1215.67, wave_lyb=1026., method='rebin', verbose=False):
    ''' identify wavelengths with LyA absorption based on a reconstructed spectra.
    '''
    w_lya = wave_lya * (1. + zobs) # observed wavelength of LyA
    w_lyb = wave_lyb * (1. + zobs) # observed wavelength of LyB

    if method == 'rebin':
        # identify LyA regions after rebinning the observed spectra to 4A
        w_coarse, fobs_coarse, iobs_coarse, fmod_coarse = rebin(wobs, fobs, iobs, zobs, wmod, fmod, 
                method='fixed', wmax=w_lya, verbose=verbose)
    elif method == 'snr_rebin': 
        # identify LyA regions after rebinning the observed spectra based on SNR
        w_coarse, fobs_coarse, iobs_coarse, fmod_coarse = rebin(wobs, fobs, iobs, zobs, wmod, fmod, 
                method='uniform', wmax=w_lya, verbose=verbose)
    else:
        raise NotImplementedError
    
    iobs_sigma = np.full_like(iobs_coarse, np.inf)
    valid_ivar = iobs_coarse > 0
    iobs_sigma[valid_ivar] = iobs_coarse[valid_ivar]**-0.5

    # above LyA 
    is_absorb_coarse = np.zeros(len(fobs_coarse)).astype(bool)
    above_lya = (w_coarse[:-1] > w_lya) 
    is_absorb_coarse[above_lya] = (fmod_coarse[above_lya] - fobs_coarse[above_lya] > 3 * iobs_sigma[above_lya])
    
    # LyA range
    lya_range = (w_coarse[:-1] > w_lyb) & (w_coarse[1:] < w_lya)
    is_absorb_coarse[lya_range] = (fmod_coarse[lya_range] - fobs_coarse[lya_range] > sigma_lya * iobs_sigma[lya_range])

    # LyB range 
    lyb_range = (w_coarse[:-1] < w_lyb)
    is_absorb_coarse[lyb_range] = (fmod_coarse[lyb_range] - fobs_coarse[lyb_range] > sigma_lyb * iobs_sigma[lyb_range])

    i_coarse = np.digitize(wobs, w_coarse, right=False) - 1
    
    is_absorb = np.zeros(len(wobs)).astype(bool)  
    is_absorb = is_absorb_coarse[i_coarse]
    return is_absorb


def rebin(wobs, fobs, iobs, zobs, wmod, fmod, method='uniform', wmax=None, verbose=True):
    if method == 'fixed': 
        w_coarse = wrebin_fixed(wobs, fobs, iobs, zobs, wmax=wmax)
    elif method == 'uniform': 
        w_coarse = wrebin_uniform(wobs, fobs, iobs, zobs, wmax=wmax)
    else: 
        raise NotImplementedError 
    if verbose: print('rebinning to Nbin = %i' % len(w_coarse))

    # observed flux and ivar coarse binned
    fobs_coarse = np.zeros(len(w_coarse)-1)
    iobs_coarse = np.zeros(len(w_coarse)-1)
    fobs_coarse[1:-1] = U.trapz_rebin(wobs, fobs, edges=w_coarse[1:-1])
    iobs_coarse[1:-1] = U.trapz_rebin(wobs, iobs/np.gradient(wobs), 
                                         edges=w_coarse[1:-1])

    # deal with edges (assume uniform binning for observed anad model spectra)
    wlim = (wobs < w_coarse[1])
    fobs_coarse[0] = np.sum(np.diff(wobs)[0] * fobs[wlim])/(w_coarse[1] - w_coarse[0])
    iobs_coarse[0] = np.sum(np.diff(wobs)[0] * (iobs/np.gradient(wobs))[wlim])/(w_coarse[1] - w_coarse[0])
    wlim = (wobs > w_coarse[-2])
    fobs_coarse[-1] = np.sum(np.diff(wobs)[0] * fobs[wlim])/(w_coarse[-1] - w_coarse[-2])
    iobs_coarse[-1] = np.sum(np.diff(wobs)[0] * (iobs/np.gradient(wobs))[wlim])/(w_coarse[-1] - w_coarse[-2])
    # rescale ivar
    iobs_coarse *= np.diff(w_coarse)
    
    # reconstructed flux coarse binned
    fmod_coarse = np.zeros(len(w_coarse)-1)
    try:
        fmod_coarse[1:-1] = U.trapz_rebin(wmod, fmod, edges=w_coarse[1:-1])
    except ValueError:
        # High-redshift spectra can put the observed grid slightly outside the
        # reconstruction range. Ignore those edge bins for absorption clipping.
        wmid = 0.5 * (w_coarse[:-1] + w_coarse[1:])
        fmod_coarse = np.interp(wmid, wmod, fmod, left=np.nan, right=np.nan)
    
    wlim = (w_coarse[0] <= wmod) & (wmod < w_coarse[1])
    fmod_coarse[-1] = np.sum(np.diff(wmod)[0] * fmod[wlim])/(w_coarse[1] - w_coarse[0])
    wlim = (w_coarse[-2] <= wmod) & (wmod < w_coarse[-1])
    fmod_coarse[-1] = np.sum(np.diff(wmod)[0] * fmod[wlim])/(w_coarse[-1] - w_coarse[-2])
    return w_coarse, fobs_coarse, iobs_coarse, fmod_coarse


def wrebin_fixed(wobs, fobs, iobs, zobs, wmax=None):
    ''' uniform rebinning where binsize is fixed to 4A 
    '''
    dw = 4.0

    # rebinning 
    Nbin = int(((wobs[-1] - wobs[0]) + 0.8)/dw)
    w_coarse = np.linspace(wobs[0]-0.4, wobs[-1]+0.4, Nbin)
    return w_coarse


def wrebin_uniform(wobs, fobs, iobs, zobs, wmax=None):
    ''' uniform rebinning where binsize is scaled by SNR over lambda < wmax  
    '''
    # number of bins based on overall snr
    snr = fobs * iobs**0.5

    # scale resolution by SNR of the spectra below LyA
    # the scaling is set so that at SNR = 1, each spectral element is 8A and the minimum 
    # bin width is 4A. 
    valid_snr = np.isfinite(snr) & (snr > 0) & (wobs < wmax)
    median_snr = np.median(snr[valid_snr]) if np.any(valid_snr) else 0.0
    dw = 16.0 if median_snr <= 0 else np.clip(8.0 / median_snr, 4.0, 16.)

    # coarse binning
    Nbin = int(((wobs[-1] - wobs[0]) + 0.8)/dw)
    w_coarse = np.linspace(wobs[0]-0.4, wobs[-1]+0.4, Nbin)
    return w_coarse

