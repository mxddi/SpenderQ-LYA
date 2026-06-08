"""
This program creates ML-reconstructed unabsorbed continuua for SDSS quasar spectra 
using a modified version of SpenderQ (located in SpenderQ.py). It also creates 
ratios and difference plots for any two spectra.

@author: Madaly
"""
 
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({"axes.labelsize": 14, "axes.titlesize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11, "figure.titlesize": 14})
import numpy as np
import torch
from astropy.io import fits

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spenderq import util as U
from spenderq.desi_qso import DESI
from spenderq.spenderq import SpenderQ

########################################################################################################################################

#CASE_DIR = Path("dat/Dense_Lya")
CASE_DIR = Path("dat/Variability/NV")
#CSV_PATH = CASE_DIR / "J2318_list_fits.csv"
CSV_PATH = CASE_DIR / "NV_variability_list_fits.csv"
SHOW_PLOT = False # Set to True to show plots (note: multiple plots are generated for each quasar, set to False for large directories)

########################################################################################################################################

MODEL_NAME = "qso.dr1.hiz"
# OUTPUT_DIR, RECON_NORM_RATIOS_DIR, and FLUX_OVER_RECON_DIR are set for each quasar at runtime
OUTPUT_DIR = CASE_DIR / "spenderq_analysis"
RECON_NORM_RATIOS_DIR = CASE_DIR / "recon_norm_ratios"
FLUX_OVER_RECON_DIR = CASE_DIR / "flux_over_recon"

# Rest wavelengths
LYA = 1215.67
LYB = 1026.00
ION_LINES = {
    "N V": (1238.82, 1242.80),
    "Si IV": (1393.755, 1402.770),
    "C IV": (1548.1950, 1550.770),
}

# Wavelength limits for plotting
RATIO_XLIM_LOW = 3500.0
RATIO_XLIM_HIGH = 6000.0


def get_column(data, *names):
    columns = {name.lower(): name for name in data.names}
    for name in names:
        if name.lower() in columns:
            return np.asarray(data[columns[name.lower()]], dtype=float)
    return None


def load_sdss_like_spectrum(path, z_qso):
    """Load an SDSS spectrum and return observed wavelength, flux, ivar."""
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

        loglam = get_column(hdu, "loglam")
        wave = 10.0**loglam if loglam is not None else get_column(
            hdu, "wave", "wavelength", "lambda"
        )
        flux = get_column(hdu, "flux")
        ivar = get_column(hdu, "ivar", "inverse_variance")
        err = get_column(hdu, "err", "error", "sigma")

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
    """Normalize near a clean continuum window."""
    wave_rest = wave_obs / (1.0 + z_qso)
    windows = [(1445.0, 1455.0), (1700.0, 1705.0)]

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
    good &= ivar > 0 # Check values are finite and that ivar > 0

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


def to_rest_frame(wave_obs, flux, ivar, z_qso):
    """Convert observed-frame arrays to rest frame by dividing wavelengths by (1+z).

    Returns (wave_rest, flux, ivar) — flux and ivar are unchanged, only the
    wavelength axis shifts.
    """
    wave_rest = wave_obs / (1.0 + z_qso)
    return wave_rest, flux.copy(), ivar.copy()


def rescale_continuum_to_flux(wave_obs, flux, continuum, ivar, z_qso):
    """Rescale continuum so its median matches the observed flux at the normalization window.

    This removes the small shift between SpenderQ's normalization and the SDSS normalization, 
    setting the continuum-normalized ratio to ~1 at the normalization window
    (1445-1455 Å restframe).
    """
    wave_rest = wave_obs / (1.0 + z_qso)
    windows = [(1445.0, 1455.0), (1700.0, 1705.0), (1275.0, 1290.0)]
    scale = np.nan
    for lo, hi in windows:
        m = (wave_rest > lo) & (wave_rest < hi) & np.isfinite(flux) & np.isfinite(continuum) & (ivar > 0) & (continuum > 0)
        if m.sum() > 5:
            scale = np.nanmedian(flux[m]) / np.nanmedian(continuum[m])
            break
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return continuum * scale, scale


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
    continuum, cont_scale = rescale_continuum_to_flux(wave_obs, flux, continuum, ivar, z_qso)
    cont_norm_ratio = np.divide(
        flux,
        continuum,
        out=np.full_like(flux, np.nan, dtype=float),
        where=np.isfinite(continuum) & (continuum != 0) & (ivar > 0),
    )

    return (
        np.asarray(recon[0]) * cont_scale,
        continuum,
        cont_norm_ratio,
        np.asarray(weight_after_lya_clip[0]),
        np.asarray(spenderq.wave_recon()),
    )



def plot_continuum_over_spectrum_restframe(path, wave_obs, flux, continuum, ivar, z_qso):
    """Plot continuum/flux in rest frame. ratio > 1 indicates absorption."""
    wave_rest, flux_rest, _ = to_rest_frame(wave_obs, flux, ivar, z_qso)

    ratio = np.divide(
        flux_rest,
        continuum,
        out=np.full_like(continuum, np.nan, dtype=float),
        where=np.isfinite(continuum) & (continuum != 0) & np.isfinite(flux_rest) & (ivar > 0), # Skip points with infinite values or no ivar
    )

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(wave_rest, ratio, color="black", lw=0.8)
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8, label="spectrum = continuum")
    ax.axhline(0.0, color="0.8", ls=":", lw=0.6)

    ax.set_xlabel("rest-frame wavelength [Å]")
    ax.set_ylabel("observed flux / SpenderQ continuum")
    ax.set_ylim(-0.2, 2.0)
    ax.set_xlim(1000.0, 1700.0)

    ax.axvline(LYA, color="tab:purple", ls="--", lw=0.9, label="Ly-alpha")
    for ion, waves in ION_LINES.items():
        for w in waves:
            ax.axvline(w, color="tab:red", ls=":", lw=0.9)
        ax.text(waves[0], ax.get_ylim()[1] * 0.97, ion, color="tab:red", fontsize=14, va="top")

    ax.legend(fontsize=14)
    outpath = OUTPUT_DIR / f"{path.stem}_continuum_over_spectrum_restframe.png"
    fig.suptitle(f"{path.name}: observed flux / SpenderQ continuum (rest frame)", fontsize=14)
    fig.savefig(outpath, dpi=200)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
    return outpath


def get_mjd(path):
    """Get the MJD from a filename like spec-PLATE-MJD-FIBER.fits."""
    parts = Path(path).stem.split("-")
    for part in parts:
        if len(part) == 5 and part.isdigit():
            return part


def save_flux_over_recon_txt(path, wave_obs, cont_norm_ratio):
    """Save flux / SpenderQ continuum as an observed-frame text file.

    Output: FLUX_OVER_RECON_DIR/{MJD}_over_recon.txt
    Columns: observed_wavelength, flux_over_recon, err
    """
    mjd = get_mjd(path)
    outpath = FLUX_OVER_RECON_DIR / f"{mjd}_over_recon.txt"
    valid = np.isfinite(wave_obs) & np.isfinite(cont_norm_ratio)
    np.savetxt(
        outpath,
        np.column_stack([wave_obs[valid], cont_norm_ratio[valid], np.zeros(valid.sum())]),
        fmt="%.6f",
        header="wave_obs  flux_over_recon  err",
    )
    return outpath


def save_cont_norm_ratio_txt(path_a, path_b, wave_obs, cont_norm_ratio_a, cont_norm_ratio_b, z_qso):
    """Save (flux/cont_A)/(flux/cont_B) and the reverse as rest-frame text files.

    Files are named <MJD_A>_over_<MJD_B>recon.txt and the reverse,
    saved to RECON_NORM_RATIOS_DIR with columns: rest_wavelength, ratio, err.
    """
    mjd_a = get_mjd(path_a)
    mjd_b = get_mjd(path_b)

    ratio_ab = np.divide(
        cont_norm_ratio_a,
        cont_norm_ratio_b,
        out=np.full_like(cont_norm_ratio_a, np.nan, dtype=float),
        where=np.isfinite(cont_norm_ratio_a) & np.isfinite(cont_norm_ratio_b) & (cont_norm_ratio_b != 0),
    )
    ratio_ba = np.divide(
        cont_norm_ratio_b,
        cont_norm_ratio_a,
        out=np.full_like(cont_norm_ratio_b, np.nan, dtype=float),
        where=np.isfinite(cont_norm_ratio_a) & np.isfinite(cont_norm_ratio_b) & (cont_norm_ratio_a != 0),
    )

    saved = []
    for mjd_num, mjd_den, ratio in [
        (mjd_a, mjd_b, ratio_ab),
        (mjd_b, mjd_a, ratio_ba),
    ]:
        valid = np.isfinite(wave_obs) & np.isfinite(ratio)
        outpath = RECON_NORM_RATIOS_DIR / f"{mjd_num}_over_{mjd_den}_recon.txt"
        np.savetxt(
            outpath,
            np.column_stack([wave_obs[valid], ratio[valid], np.full(valid.sum(), np.nan)]),
            fmt="%.6f",
        )
        saved.append(outpath)
    return saved
    

def plot_cont_norm_ratio_observed(path_a, path_b, wave_obs, cont_norm_ratio_a, cont_norm_ratio_b, z_qso):
    """Plot (flux1/cont1) / (flux2/cont2) and the reverse in observed wavelength."""
    label_a = path_a.stem
    label_b = path_b.stem
    mjd_a = get_mjd(path_a)
    mjd_b = get_mjd(path_b)

    ratio_ab = np.divide(
        cont_norm_ratio_a,
        cont_norm_ratio_b,
        out=np.full_like(cont_norm_ratio_a, np.nan, dtype=float),
        where=np.isfinite(cont_norm_ratio_a) & np.isfinite(cont_norm_ratio_b) & (cont_norm_ratio_b != 0),
    )
    ratio_ba = np.divide(
        cont_norm_ratio_b,
        cont_norm_ratio_a,
        out=np.full_like(cont_norm_ratio_b, np.nan, dtype=float),
        where=np.isfinite(cont_norm_ratio_a) & np.isfinite(cont_norm_ratio_b) & (cont_norm_ratio_a != 0),
    )

    lya_obs = LYA * (1.0 + z_qso)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    for ax, ratio, num, den in [
        (axes[0], ratio_ab, mjd_a, mjd_b),
        (axes[1], ratio_ba, mjd_b, mjd_a),
    ]:
        ax.plot(wave_obs, ratio, color="black", lw=0.8)
        ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
        ax.set_ylabel(f"MJD {num} / MJD {den}")
        ax.set_ylim(-0.2, 4.0)
        ax.axvline(lya_obs, color="tab:purple", ls="--", lw=0.9)
        for ion, waves in ION_LINES.items():
            for w in waves:
                ax.axvline(w * (1.0 + z_qso), color="tab:red", ls=":", lw=0.9)
        axes[0].text(
            ION_LINES["N V"][0] * (1.0 + z_qso), axes[0].get_ylim()[1],
            "N V", color="tab:red", fontsize=14, va="top"
        )
        axes[0].text(
            ION_LINES["C IV"][0] * (1.0 + z_qso), axes[0].get_ylim()[1],
            "C IV", color="tab:red", fontsize=14, va="top"
        )

    axes[1].set_xlabel("observed wavelength [Å]")
    axes[1].set_xlim(3800.0, 6000.0)

    outpath = OUTPUT_DIR / f"{label_a}_vs_{label_b}_cont_norm_ratio_observed.png"
    fig.suptitle(
        f"(flux/cont) ratio in observed frame: {label_a} vs {label_b}", fontsize=14
    )
    fig.savefig(outpath, dpi=200)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
    return outpath


def plot_cont_norm_ratio_difference_observed(path_a, path_b, wave_obs, cont_norm_ratio_a, cont_norm_ratio_b, z_qso):
    """Plot (flux1/cont1) - (flux2/cont2) and the reverse in observed wavelength (3800-6000 Å)."""
    label_a = path_a.stem
    label_b = path_b.stem
    mjd_a = get_mjd(path_a)
    mjd_b = get_mjd(path_b)

    diff_ab = cont_norm_ratio_a - cont_norm_ratio_b
    diff_ba = cont_norm_ratio_b - cont_norm_ratio_a

    lya_obs = LYA * (1.0 + z_qso)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    for ax, diff, num, den in [
        (axes[0], diff_ab, mjd_a, mjd_b),
        (axes[1], diff_ba, mjd_b, mjd_a),
    ]:
        ax.plot(wave_obs, diff, color="black", lw=0.8)
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.set_ylabel(f"MJD {num} − MJD {den}")
        ax.set_ylim(-1.5, 1.5)
        ax.axvline(lya_obs, color="tab:purple", ls="--", lw=0.9)
        for ion, waves in ION_LINES.items():
            for w in waves:
                ax.axvline(w * (1.0 + z_qso), color="tab:red", ls=":", lw=0.9)

    axes[0].text(ION_LINES["N V"][0] * (1.0 + z_qso), axes[0].get_ylim()[1], "N V", color="tab:red", fontsize=14, va="top")
    axes[0].text(ION_LINES["C IV"][0] * (1.0 + z_qso), axes[0].get_ylim()[1], "C IV", color="tab:red", fontsize=14, va="top")
    axes[1].set_xlabel("observed wavelength [Å]")
    axes[1].set_xlim(3800.0, 6000.0)

    outpath = OUTPUT_DIR / f"{label_a}_vs_{label_b}_cont_norm_ratio_difference_observed.png"
    fig.suptitle(
        f"(flux/cont) difference in observed frame: {label_a} vs {label_b}", fontsize=14
    )
    fig.savefig(outpath, dpi=200)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
    return outpath


def plot_analysis(path, wave_obs, flux, continuum, weight, weight_clipped, z_qso):
    wave_rest = wave_obs / (1.0 + z_qso)
    lya_forest = (wave_rest > LYB) & (wave_rest < LYA)
    lya_removed = lya_forest & (weight > 0) & (weight_clipped == 0)

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    ax.plot(wave_rest, flux, color="0.25", lw=0.7, label="spectrum")
    ax.plot(wave_rest, continuum, color="tab:blue", lw=1.2, label="SpenderQ continuum")
    ax.scatter(
        wave_rest[lya_removed],
        flux[lya_removed],
        s=5,
        color="tab:orange",
        alpha=0.5,
        label="masked Ly-alpha forest",
    )
    wave_lo, wave_hi = 1000.0, 1700.0
    ax.set_xlim(wave_lo, wave_hi)
    ax.set_ylim(*np.nanpercentile(flux[(wave_rest > wave_lo) & (wave_rest < wave_hi)], [1, 99]))
    ax.set_ylabel("normalized flux")
    ax.set_xlabel("rest-frame wavelength [Å]")
    ax.legend(loc="upper right", fontsize=14)

    ax.axvline(LYA, color="tab:purple", ls="--", lw=0.9, label="Ly-alpha")
    for ion, waves in ION_LINES.items():
        for wave in waves:
            ax.axvline(wave, color="tab:red", ls=":", lw=0.9)
        ax.text(waves[0], ax.get_ylim()[1], ion, color="tab:red", fontsize=14, va="top")

    outpath = OUTPUT_DIR / f"{path.stem}_spenderq_ehvo.png"
    fig.suptitle(f"{path.name}: SpenderQ continuum", fontsize=14)
    fig.savefig(outpath, dpi=200)
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    return outpath, int(np.count_nonzero(lya_removed))


def quasar_dir(quasar_name):
    """Return the directory stem: everything before the '+' in the quasar name."""
    idx = quasar_name.find("+")
    return quasar_name[:idx] if idx != -1 else quasar_name


def analyze_file(path, z_qso):
    wave_obs, flux, ivar = load_sdss_like_spectrum(path, z_qso)
    flux, ivar, norm = normalize_spectrum(wave_obs, flux, ivar, z_qso)
    wave_grid, flux_grid, ivar_grid = resample_to_spenderq_grid(wave_obs, flux, ivar)
    recon_rest, continuum_grid, cont_norm_ratio, weight_clipped, wave_rest = run_spenderq(
        wave_grid, flux_grid, ivar_grid, z_qso
    )

    analysis_path, n_lya_removed = plot_analysis(
        path,
        wave_grid,
        flux_grid,
        continuum_grid,
        ivar_grid,
        weight_clipped,
        z_qso,
    )
    ratio_path = plot_continuum_over_spectrum_restframe(
        path, wave_grid, flux_grid, continuum_grid, ivar_grid, z_qso
    )
    flux_recon_path = save_flux_over_recon_txt(path, wave_grid, cont_norm_ratio)
    print(f"{path.name}: normalization={norm:.4g}")
    print(f"Saved {analysis_path} ({n_lya_removed} Ly-alpha forest pixels masked)")
    print(f"Saved {ratio_path}")
    print(f"Saved {flux_recon_path}")

    return {
        "path": path,
        "wave_grid": wave_grid,
        "continuum_grid": continuum_grid,
        "cont_norm_ratio": cont_norm_ratio,
    }


def read_csv(csv_path):
    """Return list of dicts with keys: name, obs1, obs2, redshift."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "name":     row["Quasar Name"].strip(),
                "obs1":     row["Observation 1"].strip(),
                "obs2":     row["Observation 2"].strip(),
                "redshift": float(row["Redshift"].strip()),
            })
    return rows


def _run_quasar(quasar_name, obs1_filename, obs2_filename, z_qso, CASE_DIR):
    """Run the full SpenderQ analysis for one quasar pair and save all outputs."""
    global OUTPUT_DIR, RECON_NORM_RATIOS_DIR

    qso_prefix = quasar_dir(quasar_name)
    qso_dir = CASE_DIR / qso_prefix
    OUTPUT_DIR = qso_dir / "spenderq_analysis"
    RECON_NORM_RATIOS_DIR = CASE_DIR / "recon_norm_ratios"
    FLUX_OVER_RECON_DIR = CASE_DIR / "flux_over_recon"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RECON_NORM_RATIOS_DIR.mkdir(parents=True, exist_ok=True)
    FLUX_OVER_RECON_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Quasar: {quasar_name}  z={z_qso}")
    print(f"Dir:    {qso_dir}")

    reconstructions = []
    for filename in (obs1_filename, obs2_filename):
        fits_path = qso_dir / filename
        if not fits_path.exists():
            print(f"  [skip] FITS not found: {fits_path}")
            continue
        result = analyze_file(fits_path, z_qso)
        if result is not None:
            reconstructions.append(result)

    if len(reconstructions) < 2:
        print(f"  [skip] need 2 spectra, got {len(reconstructions)} — skipping cross-file plots")
        return None

    a, b = reconstructions[0], reconstructions[1]
    wave = a["wave_grid"]
    cont_a = a["continuum_grid"]
    cont_b = np.interp(wave, b["wave_grid"], b["continuum_grid"], left=np.nan, right=np.nan)
    in_range = (wave >= RATIO_XLIM_LOW) & (wave <= RATIO_XLIM_HIGH)

    # Continuum ratio text files
    for label_num, label_den, ratio in [
        (a["path"].stem, b["path"].stem,
         np.divide(cont_a, cont_b, out=np.full_like(cont_a, np.nan),
                   where=np.isfinite(cont_a) & np.isfinite(cont_b) & (cont_b != 0))),
        (b["path"].stem, a["path"].stem,
         np.divide(cont_b, cont_a, out=np.full_like(cont_b, np.nan),
                   where=np.isfinite(cont_a) & np.isfinite(cont_b) & (cont_a != 0))),
    ]:
        valid = in_range & np.isfinite(ratio)
        outpath = OUTPUT_DIR / f"{label_num}_vs_{label_den}_ratio_norm.txt"
        np.savetxt(outpath,
                   np.column_stack([wave[valid], ratio[valid], np.full(valid.sum(), np.nan)]),
                   fmt="%.6f")
        print(f"Saved {outpath}")

    # Continuum ratio panels figure
    ratio_ab = np.divide(cont_a, cont_b, out=np.full_like(cont_a, np.nan),
                         where=np.isfinite(cont_a) & np.isfinite(cont_b) & (cont_b != 0))
    ratio_ba = np.divide(cont_b, cont_a, out=np.full_like(cont_b, np.nan),
                         where=np.isfinite(cont_a) & np.isfinite(cont_b) & (cont_a != 0))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    axes[0].plot(wave[in_range], ratio_ab[in_range], color="black", lw=0.9)
    axes[0].axhline(1.0, color="0.5", ls="--", lw=0.8)
    axes[0].set_ylabel(f"{a['path'].stem}/{b['path'].stem}")
    axes[1].plot(wave[in_range], ratio_ba[in_range], color="black", lw=0.9)
    axes[1].axhline(1.0, color="0.5", ls="--", lw=0.8)
    axes[1].set_ylabel(f"{b['path'].stem}/{a['path'].stem}")
    axes[1].set_xlabel("observed wavelength [Å]")
    axes[1].set_xlim(RATIO_XLIM_LOW, RATIO_XLIM_HIGH)
    ratio_png = OUTPUT_DIR / f"{a['path'].stem}_vs_{b['path'].stem}_continuum_ratio_panels.png"
    fig.suptitle(f"SpenderQ continuum ratios: {a['path'].stem} vs {b['path'].stem}", fontsize=14)
    fig.savefig(ratio_png, dpi=200)
    if not SHOW_PLOT:
        plt.close(fig)
    print(f"Saved {ratio_png}")

    trans_ratio_png = plot_cont_norm_ratio_observed(
        a["path"], b["path"], a["wave_grid"],
        a["cont_norm_ratio"], b["cont_norm_ratio"], z_qso,
    )
    print(f"Saved {trans_ratio_png}")

    trans_diff_png = plot_cont_norm_ratio_difference_observed(
        a["path"], b["path"], a["wave_grid"],
        a["cont_norm_ratio"], b["cont_norm_ratio"], z_qso,
    )
    print(f"Saved {trans_diff_png}")

    ratio_txts = save_cont_norm_ratio_txt(
        a["path"], b["path"], a["wave_grid"],
        a["cont_norm_ratio"], b["cont_norm_ratio"], z_qso,
    )
    for p in ratio_txts:
        print(f"Saved {p}")

    return ratio_txts, z_qso


def save_summary_csv(case_dir, entries):
    """Write a summary CSV of all recon ratio files and their redshifts.

    Parameters
    ----------
    case_dir : Path
        Root directory
    entries : list of (Path, float)
        Each element is (ratio_txt_path, z_qso) collected from _run_quasar.
    """
    case_name = case_dir.resolve().name
    out_csv = case_dir / f"{case_name}_recon_ratios_norm.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["NORM SPECTRA FILE NAME", "REDSHIFT", "CALCULATED SNR",
                         "NEEDS RECALCULATION", "Masked Regions"])
        for txt_path, z in entries:
            writer.writerow([txt_path.name, z, 0, "N", "[]"])
    print(f"\nSaved summary CSV: {out_csv}")
    return out_csv


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    quasars = read_csv(CSV_PATH)
    print(f"Found {len(quasars)} quasar(s) in {CSV_PATH}")

    summary_entries = []
    for q in quasars:
        result = _run_quasar(q["name"], q["obs1"], q["obs2"], q["redshift"], CASE_DIR)
        if result is not None:
            ratio_txts, z = result
            for p in ratio_txts:
                summary_entries.append((p, z))

    if summary_entries:
        save_summary_csv(CASE_DIR, summary_entries)