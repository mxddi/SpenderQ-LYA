import numpy as np
from astropy.io import fits
import os

specdirec = "src/spenderq/dat/J2318/"

def txt_to_fits_restframe(filename):
    """
    Converts rest-frame SDSS .txt spectra to .fits.
    Sets Z=0 in header to prevent Spender from re-shifting.
    """

    clean_name = filename.replace("-dered.dr16", "").replace(".txt", "")
    parts = clean_name.split("-")
    plate, mjd, fiber = parts[0], parts[1], parts[2]
    
    data = np.loadtxt(os.path.join(specdirec, filename))
    wave, flux, err = data[:, 0], data[:, 1], data[:, 2]
    
    # Calculate inverse variance (ivar)
    ivar = np.zeros_like(err)
    valid_mask = (err > 0) & (~np.isnan(err))
    ivar[valid_mask] = 1.0 / (err[valid_mask]**2)

    # TODO: Remove if normalizd already. Normalization (Directly at 1450A since it's rest-frame)
    mask_1450 = (wave > 1445) & (wave < 1455)
    if np.any(mask_1450):
        norm_factor = np.median(flux[mask_1450])
    else:
        norm_factor = np.median(flux) # Fallback to global median

    normalized_flux = flux / norm_factor
    normalized_ivar = ivar * (norm_factor**2)
    
    # Create FITS
    primary_hdu = fits.PrimaryHDU()
    # SET TO 0. SpenderQ shifts by 1/(1+z). If already in restframe, z must be 0.
    primary_hdu.header['Z'] = 0.0  
    
    cols = fits.ColDefs([
        fits.Column(name='loglam', format='E', array=np.log10(wave)),
        fits.Column(name='flux', format='E', array=normalized_flux),
        fits.Column(name='ivar', format='E', array=normalized_ivar)
    ])
    
    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.name = 'COADD'
    
    # Save FITS
    out_fits_name = f"{specdirec}spec-{plate}-{mjd}-{fiber}.fits"
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(out_fits_name, overwrite=True)
    
    print(f"Saved rest-frame FITS: {out_fits_name}")
    return out_fits_name

# Use function
files = ["6138-57328-0746-dered.dr16", "6138-59188-0746-dered.dr16", "6138-60251-0746-dered.dr16"]
for f in files:
    txt_to_fits_restframe(f)

# verify by loading one of the FITS files
from astropy.io import fits
fits_file = "src/spenderq/dat/J2318/spec-6138-57328-0746.fits"
with fits.open(fits_file) as hdul:
    print(hdul.info())
    print(hdul[0].header)
    print(hdul[1].columns)



import matplotlib.pyplot as plt

# Check the file is correctly converted by plotting
def check_fits_plot(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul['COADD'].data
        wave = 10**data['loglam']
        flux = data['flux']
        
        plt.figure(figsize=(10, 4))
        plt.step(wave[:1200], flux[:1200], where='mid', color='black', lw=0.5, label='Normalized Flux')
        plt.axhline(1, color='red', ls='--', label='Normalization Level (1450A)')
        plt.xlabel('Rest-frame Wavelength [A]')
        plt.ylabel('Normalized Flux')
        plt.title(f"Checking: {fits_path}")
        plt.legend()
        plt.show()

# Run plot check for a file
check_fits_plot(f"{specdirec}spec-6138-57328-0746.fits")
