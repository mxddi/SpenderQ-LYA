import numpy as np
from scipy.ndimage import median_filter

def clean_lyalpha_residuals(wave_rest, flux_obs, flux_spender_continuum):
    """
    Removes sharp LyA forest lines from overlapping broad outflow troughs.
    
    Parameters:
    wave_rest (array): Rest-frame wavelength array
    flux_obs (array): Your original observed flux
    flux_spender_continuum (array): The smooth continuum predicted by SpenderQ
    """
    # Step 1: Normalize the spectrum
    # This flattens the continuum to ~1.0 and isolates all absorption features
    flux_norm = flux_obs / flux_spender_continuum
    
    # Step 2: Use a rolling median filter to detect feature widths
    # A window of ~25 pixels usually corresponds to ~500-1000 km/s depending on resolution.
    # The median filter ignores sharp spikes but follows the broad wind valleys.
    smooth_features = median_filter(flux_norm, size=25)
    
    # Step 3: Identify the sharp LyA forest pixels
    # If the real data drops significantly below the smooth median trend, it's a forest line.
    forest_threshold = 0.08  # Adjust based on data noise
    is_forest = (smooth_features - flux_norm) > forest_threshold
    
    # Create the clean mask (True for good data, False for LyA contamination)
    clean_mask = ~is_forest
    
    # Step 4: Linearly interpolate across the LyA gaps
    # We use the 'good' pixels to guess the values of the 'bad' forest pixels
    flux_patched = np.copy(flux_norm)
    flux_patched[is_forest] = np.interp(
        wave_rest[is_forest],    # Where we need to guess data
        wave_rest[clean_mask],   # X-coordinates of good data
        flux_norm[clean_mask]    # Y-coordinates of good data
    )
    
    return flux_norm, flux_patched, clean_mask

# --- How to visualize the result ---
# flux_norm: Has your broad troughs AND the jagged forest spikes.
# flux_patched: The final product! Smooth broad troughs with the forest erased.