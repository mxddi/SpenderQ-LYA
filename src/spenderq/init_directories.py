"""
init_directories.py

Reads a variability CSV (format: Quasar Name, Observation 1, Observation 2, Redshift),
creates one subdirectory per quasar, and downloads the two SDSS DR16 FITS files into that subdirectory.

Usage:
    python init_directories.py [CSV_PATH] [BASE_DIR]

    CSV_PATH   path to the CSV file (default: dat/Variability/NV/NV_variability_list_fits.csv)
    BASE_DIR   root directory to create quasar subdirectories in
                (default: same directory as the CSV file)

Example:
    python init_directories.py dat/Variability/NV/NV_variability_list_fits.csv dat/Variability/NV
"""

import csv
import sys
import urllib.request
import urllib.error
from pathlib import Path

# SDSS DR16 SAS base URLs tried in order for each spectrum.
# Older SDSS-I/II plates (roughly < 3510) live under sdss/spectro/redux/26;
# BOSS/eBOSS plates are under eboss/spectro/redux/v5_13_0.
# Tries both so you do not need to know which survey a plate belongs to.

SDSS_BASE_URLS = [
    "https://data.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0/spectra/full",
    "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/full",
]


def parse_fits_filename(filename):
    """Return (plate, mjd, fiber) ints from 'spec-PLATE-MJD-FIBER.fits'."""
    stem = Path(filename).stem          # e.g. spec-0948-52428-0370
    parts = stem.split("-")             # ['spec', '0948', '52428', '0370']
    if len(parts) != 4 or parts[0] != "spec":
        raise ValueError(f"Unexpected FITS filename format: {filename!r}")
    plate = int(parts[1])
    mjd   = int(parts[2])
    fiber = int(parts[3])
    return plate, mjd, fiber


def sdss_download_urls(filename):
    """Yield candidate SAS URLs for a given spec filename."""
    plate, mjd, fiber = parse_fits_filename(filename)
    fname = f"spec-{plate:04d}-{mjd:05d}-{fiber:04d}.fits"
    plate_dir = f"{plate:04d}"
    for base in SDSS_BASE_URLS:
        yield f"{base}/{plate_dir}/{fname}"


def download_fits(filename, dest_dir):
    """Download *filename* from SDSS SAS into dest_dir. Returns the local path."""
    dest = dest_dir / Path(filename).name
    if dest.exists():
        print(f"  [skip] already exists: {dest}")
        return dest

    last_error = None
    for url in sdss_download_urls(filename):
        try:
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"  -> saved to {dest}")
            return dest
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                print(f"  [404] not found at {url}, trying next …")
                continue
            raise
        except Exception as exc:
            last_error = exc
            print(f"  [error] {exc}, trying next …")
            continue

    raise RuntimeError(
        f"Could not download {filename!r} from any known SAS location. "
        f"Last error: {last_error}"
    )


def quasar_dir_name(quasar_name):
    """Return the directory name: everything up to (not including) the '+' sign."""
    idx = quasar_name.find("+")
    if idx == -1:
        return quasar_name          # no '+' (e.g. negative-dec quasar — keep full name)
    return quasar_name[:idx]


def init_directories(csv_path, base_dir):
    csv_path = Path(csv_path)
    base_dir = Path(base_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    base_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    print(f"Found {len(rows)} quasar(s) in {csv_path}\n")

    for row in rows:
        name   = row["Quasar Name"].strip()
        obs1   = row["Observation 1"].strip()
        obs2   = row["Observation 2"].strip()
        z      = row["Redshift"].strip()

        dir_name = quasar_dir_name(name)
        qso_dir  = base_dir / dir_name
        qso_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'='*60}")
        print(f"Quasar : {name}  (z = {z})")
        print(f"Dir    : {qso_dir}")

        for obs in (obs1, obs2):
            try:
                download_fits(obs, qso_dir)
            except RuntimeError as exc:
                print(f"  [FAILED] {exc}")

    print(f"\nQuasar directories are nowunder: {base_dir.resolve()}")


if __name__ == "__main__":
    default_csv = Path("dat/Variability/NV/NV_variability_list_fits.csv")
    csv_arg  = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv
    base_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_arg.parent

    init_directories(csv_arg, base_arg)
