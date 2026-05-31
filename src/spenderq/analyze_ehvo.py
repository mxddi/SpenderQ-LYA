from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spenderq import util as U
from spenderq.desi_qso import DESI
from spenderq.spenderq import SpenderQ

SPECTRUM_DIR = Path("src/spenderq/dat/Variability/6122/")
FILENAMES = [
    "spec-6122-56246-0696.fits",
]

Z_QSO = 3.623 # Original observation redshift

MODEL_NAME = "qso.dr1.hiz"
OUTPUT_DIR = Path("src/spenderq/dat/Variability/6122/spenderq_analysis")
SHOW_PLOT = True

LYA = 1215.67
LYB = 1026.00
ION_LINES = {
    "N V": (1238.821, 1242.804),
    "C IV": (1548.204, 1550.781),
}


def _get_column(data, *names):
    columns = {name.lower(): name for name in data.names}
    for name in names:
        if name.lower() in columns:
            return np.asarray(data[columns[name.lower()]], dtype=float)
    return None


def load_sdss_like_spectrum(path, z_qso):
    """Load an SDSS-style spectrum and return observed wavelength, flux, ivar."""
    with fits.open(path) as hdul:
        hdu = None
        for candidate in ("COADD", 1):
            try:
                data = hdul[candidate].data
            except (KeyError, IndexError):
                continue
            if getattr(data, "names", None):
                hdu = data
                break

        if hdu is None:
            raise ValueError(f"No table HDU with spectral columns found in {path}")

        loglam = _get_column(hdu, "loglam")
        wave = 10.0**loglam if loglam is not None else _get_column(
            hdu, "wave", "wavelength", "lambda"
        )
        flux = _get_column(hdu, "flux")
        ivar = _get_column(hdu, "ivar", "inverse_variance")
        err = _get_column(hdu, "err", "error", "sigma")

        if wave is None or flux is None:
            raise ValueError(f"Missing wavelength or flux column in {path}")
        if ivar is None:
            if err is None:
                ivar = np.ones_like(flux)
            else:
                ivar = np.zeros_like(err)
                good_err = np.isfinite(err) & (err > 0)
                ivar[good_err] = 1.0 / err[good_err] ** 2

    if np.nanpercentile(wave, 95) < np.nanmin(np.asarray(DESI._wave_obs)):
        wave = wave * (1.0 + z_qso)

    return wave, flux, ivar


def normalize_spectrum(wave_obs, flux, ivar, z_qso):
    """Normalize near a relatively clean UV continuum window."""
    wave_rest = wave_obs / (1.0 + z_qso)
    windows = [(1445.0, 1455.0), (1700.0, 1705.0), (1275.0, 1290.0)]

    norm = np.nan
    for lo, hi in windows:
        use = (wave_rest > lo) & (wave_rest < hi) & np.isfinite(flux) & (ivar > 0)
        if np.count_nonzero(use) > 5:
            norm = np.nanmedian(flux[use])
            break

    if not np.isfinite(norm) or norm == 0:
        use = np.isfinite(flux) & (ivar > 0)
        norm = np.nanmedian(flux[use])

    return flux / norm, ivar * norm**2, norm


def resample_to_spenderq_grid(wave_obs, flux, ivar):
    """Interpolate a single spectrum onto SpenderQ's DESI wavelength grid."""
    wave_grid = np.asarray(DESI._wave_obs, dtype=float)
    good = np.isfinite(wave_obs) & np.isfinite(flux) & np.isfinite(ivar)
    good &= ivar > 0

    order = np.argsort(wave_obs[good])
    wave_in = wave_obs[good][order]
    flux_in = flux[good][order]
    ivar_in = ivar[good][order]

    flux_grid = np.interp(wave_grid, wave_in, flux_in, left=np.nan, right=np.nan)
    ivar_grid = np.interp(wave_grid, wave_in, ivar_in, left=0.0, right=0.0)

    bad = ~np.isfinite(flux_grid)
    flux_grid[bad] = 0.0
    ivar_grid[bad] = 0.0

    return wave_grid, flux_grid.astype(np.float32), ivar_grid.astype(np.float32)


def rebin_reconstruction_to_observed(wave_rest_model, recon_rest, wave_obs, z_qso):
    wave_model_obs = wave_rest_model * (1.0 + z_qso)
    try:
        return U.trapz_rebin(wave_model_obs, recon_rest, xnew=wave_obs)
    except ValueError:
        return np.interp(wave_obs, wave_model_obs, recon_rest, left=np.nan, right=np.nan)


def run_spenderq(wave_obs, flux, ivar, z_qso):
    spec = torch.from_numpy(flux[None, :])
    weight = torch.from_numpy(ivar[None, :])
    z = torch.tensor([z_qso], dtype=torch.float32)

    spenderq = SpenderQ(MODEL_NAME)
    weight_after_lya_clip = weight.clone()
    _, recon = spenderq.eval(spec, weight_after_lya_clip, z)

    continuum = rebin_reconstruction_to_observed(
        spenderq.wave_recon(), np.asarray(recon[0]), wave_obs, z_qso
    )
    transmission = np.divide(
        flux,
        continuum,
        out=np.full_like(flux, np.nan, dtype=float),
        where=np.isfinite(continuum) & (continuum != 0) & (ivar > 0),
    )

    return np.asarray(recon[0]), continuum, transmission, np.asarray(weight_after_lya_clip[0])


def plot_analysis(path, wave_obs, flux, continuum, transmission, weight, weight_clipped, z_qso):
    wave_rest = wave_obs / (1.0 + z_qso)
    lya_forest = (wave_rest > LYB) & (wave_rest < LYA)
    lya_removed = lya_forest & (weight > 0) & (weight_clipped == 0)
    transmission_no_lya = transmission.copy()
    transmission_no_lya[lya_removed] = np.nan

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False, constrained_layout=True)

    axes[0].plot(wave_rest, flux, color="0.25", lw=0.7, label="spectrum")
    axes[0].plot(wave_rest, continuum, color="tab:blue", lw=1.2, label="SpenderQ continuum")
    axes[0].scatter(
        wave_rest[lya_removed],
        flux[lya_removed],
        s=5,
        color="tab:orange",
        alpha=0.5,
        label="removed Ly-alpha forest pixels",
    )
    axes[0].set_xlim(1000.0, 1700.0)
    axes[0].set_ylim(*np.nanpercentile(flux[(wave_rest > 1000) & (wave_rest < 1700)], [1, 99]))
    axes[0].set_ylabel("normalized flux")
    axes[0].legend(loc="upper right", fontsize=9)

    absorption_depth = 1.0 - transmission_no_lya
    axes[1].plot(wave_rest, absorption_depth, color="black", lw=0.8)
    axes[1].axhline(0.0, color="0.5", ls="--", lw=0.8)
    axes[1].set_xlim(1150.0, 1650.0)
    axes[1].set_ylim(-0.4, 1.2)
    axes[1].set_ylabel("absorption depth")

    for ax in axes[:2]:
        ax.axvline(LYA, color="tab:purple", ls="--", lw=0.9, label="Ly-alpha")
        for ion, waves in ION_LINES.items():
            for wave in waves:
                ax.axvline(wave, color="tab:red", ls=":", lw=0.9)
            ax.text(waves[0], ax.get_ylim()[1], ion, color="tab:red", fontsize=9, va="top")

    axes[1].set_xlabel("rest-frame wavelength [Angstrom]")

    outpath = OUTPUT_DIR / f"{path.stem}_spenderq_ehvo.png"
    fig.suptitle(f"{path.name}: SpenderQ continuum with Ly-alpha forest removed", fontsize=13)
    fig.savefig(outpath, dpi=200)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    return outpath, int(np.count_nonzero(lya_removed))


def analyze_file(path, z_qso):
    wave_obs, flux, ivar = load_sdss_like_spectrum(path, z_qso)
    flux, ivar, norm = normalize_spectrum(wave_obs, flux, ivar, z_qso)
    wave_grid, flux_grid, ivar_grid = resample_to_spenderq_grid(wave_obs, flux, ivar)
    _, continuum, transmission, weight_clipped = run_spenderq(
        wave_grid, flux_grid, ivar_grid, z_qso
    )
    outpath, n_lya_removed = plot_analysis(
        path,
        wave_grid,
        flux_grid,
        continuum,
        transmission,
        ivar_grid,
        weight_clipped,
        z_qso,
    )
    print(f"{path.name}: normalization={norm:.4g}, removed Ly-alpha pixels={n_lya_removed}")
    print(f"Saved {outpath}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    for filename in FILENAMES:
        analyze_file(SPECTRUM_DIR / filename, Z_QSO)