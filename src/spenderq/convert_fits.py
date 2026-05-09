import numpy as np
from astropy.io import fits
from astropy.table import Table

def convert_txt_to_fits(txt_file, fits_output, redshift):
    # 1. Load your text data (adjust delimiter/usecols as needed)
    # Assumes columns: wavelength, flux, ivar (or error)
    data = np.loadtxt(txt_file)
    wave = data[:, 0]
    flux = data[:, 1]
    ivar = data[:, 2] # Inverse variance

    # 2. Create an HDU list
    # Primary HDU stores metadata (headers)
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['Z'] = redshift  # SpenderQ needs the redshift
    
    # 3. Create Binary Table HDU for the data
    col1 = fits.Column(name='loglam', format='E', array=np.log10(wave))
    col2 = fits.Column(name='flux', format='E', array=flux)
    col3 = fits.Column(name='ivar', format='E', array=ivar)
    
    table_hdu = fits.BinTableHDU.from_columns([col1, col2, col3])
    table_hdu.name = 'COADD' # Standard SDSS HDU name

    # 4. Save the file
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(fits_output, overwrite=True)

# Example usage
convert_txt_to_fits('spec_123.txt', 'spec-0123-55555-001.fits', redshift=2.15)
