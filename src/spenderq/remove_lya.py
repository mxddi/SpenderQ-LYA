import numpy as np
from scipy.ndimage import median_filter

def clean_lyalpha_residuals(wave_rest, flux_obs, flux_spender_continuum):
    """
    Removes sharp LyA forest lines from overlapping outflow ion absorption troughs.
    
    Parameters:
    wave_rest (array): Rest-frame wavelength array
    flux_obs (array): Your original observed flux
    flux_spender_continuum (array): The smooth continuum predicted by SpenderQ
    """
    # Normalize the spectrum. This flattens the continuum to ~1.0 and isolates all absorption features
    flux_norm = flux_obs / flux_spender_continuum
    
    # Median filter to detect absorption feature widths.
    # A window of ~25 pixels usually corresponds to ~500-1000 km/s.The median filter ignores sharp spikes but follows the broad EHVO absorption troughs.
    smooth_features = median_filter(flux_norm, size=25)
    
    # Identify the sharp LyA forest lines. If the real spectrum drops significantly below the smooth median then it's considered a LyA forest line.
    forest_threshold = 0.08  # To adjust based on data noise
    is_forest = (smooth_features - flux_norm) > forest_threshold
    
    # Create the mask. True for good data, False for LyA lines)
    clean_mask = ~is_forest
    
    # Interpolate across the LyA gaps
    # The good pixels are used to guess the values of the LyA pixels
    flux_without_lya = np.copy(flux_norm)
    flux_without_lya[is_forest] = np.interp(
        wave_rest[is_forest],    # Where we need to guess data
        wave_rest[clean_mask],   # X-coordinates of good data
        flux_norm[clean_mask]    # Y-coordinates of good data
    )
    
    return flux_norm, flux_without_lya, clean_mask

# Result should look like:
# flux_norm: Has broad troughs overlapping with LyA forest spikes
# flux_without_lya: Smooth broad troughs witout the Lya spikes