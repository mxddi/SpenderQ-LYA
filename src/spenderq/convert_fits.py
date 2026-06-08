import numpy as np
from astropy.io import fits
import os

input_dir = "dat/Dense_Lya/J2318/unnormalized/"
output_dir = "dat/Dense_Lya/J2318/"
z_qso = 2.678


def txt_to_fits_observed_frame(filename):
    """
    Convert a rest-frame SDSS text spectrum to an observed-frame FITS file.
    """

    clean_name = filename.replace("-dered.dr16", "").replace(".txt", "")
    parts = clean_name.split("-")
    plate, mjd, fiber = parts[0], parts[1], parts[2]
    
    data = np.loadtxt(os.path.join(input_dir, filename))
    wave_rest, flux, err = data[:, 0], data[:, 1], data[:, 2]
    wave_obs = wave_rest * (1.0 + z_qso)
    
    # Calculate inverse variance (ivar)
    ivar = np.zeros_like(err)
    valid_mask = (err > 0) & (~np.isnan(err))
    ivar[valid_mask] = 1.0 / (err[valid_mask]**2)

    # TODO: Remove if normalizd already. Normalization (directly at 1450A since input is rest-frame)
    mask_1450 = (wave_rest > 1445) & (wave_rest < 1455)
    if np.any(mask_1450):
        norm_factor = np.median(flux[mask_1450])
    else:
        norm_factor = np.median(flux) # Fallback to global median

    normalized_flux = flux / norm_factor
    normalized_ivar = ivar * (norm_factor**2)
    
    # Create FITS
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['Z'] = z_qso
    
    cols = fits.ColDefs([
        fits.Column(name='loglam', format='E', array=np.log10(wave_obs)),
        fits.Column(name='flux', format='E', array=normalized_flux),
        fits.Column(name='ivar', format='E', array=normalized_ivar)
    ])
    
    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.name = 'COADD'
    
    # Save FITS
    os.makedirs(output_dir, exist_ok=True)
    out_fits_name = f"{output_dir}spec-{plate}-{mjd}-{fiber}.fits"
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(out_fits_name, overwrite=True)
    
    print(f"Saved observed-frame FITS: {out_fits_name}")
    return out_fits_name

# Use function
files = ["6138-57328-0746-dered.dr16", "6138-59188-0746-dered.dr16", "6138-60251-0746-dered.dr16"]
for f in files:
    txt_to_fits_observed_frame(f)

# verify by loading one of the FITS files
fits_file = f"{output_dir}spec-6138-57328-0746.fits"
with fits.open(fits_file) as hdul:
    print(hdul.info())
    print(hdul[0].header)
    print(hdul[1].columns)

