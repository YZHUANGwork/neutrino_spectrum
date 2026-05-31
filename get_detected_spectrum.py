"""
get_detected_spectrum.py
=========================
User inputs:
    - target        : 'Xe' or 'Ar'
    - channel       : 'NR' or 'ER'
    - nu_sources    : list of neutrino source names
    - bkgd_sources  : list of background names (optional, ER only: Kr85, Rn222, nubb)
    - mode          : 'ideal'     -> sum only
                      'realistic' -> sum -> smear -> efficiency
    - detector      : required for mode='realistic'

Returns (Er, rate) ready to plot.

Example
-------
    from get_detected_spectrum import get_detected_spectrum

    # NR ideal
    Er, rate = get_detected_spectrum(
        target='Xe', channel='NR',
        nu_sources=['pp', 'Be7_861', '8B', 'hep', 'atmNu_SURF_avg'],
        mode='ideal')

    # ER realistic with backgrounds
    Er, rate = get_detected_spectrum(
        target='Ar', channel='ER',
        nu_sources=['pp', 'Be7_861', 'pep', 'N13', 'O15', 'F17'],
        bkgd_sources=['Kr85', 'Rn222', 'nubb'],
        mode='realistic', detector='ideal Ethrd1keV')
"""

import numpy as np
import os
import glob
import astropy.units as u
from scipy.interpolate import interp1d

from cross_section_to_rate import nr_rate, er_rate

_RATE_UNIT       = 1 / u.tonne / u.yr / u.keV
_BKGD_FOLDER    = 'measured_spectrum'

def _read(filepath):
    """Read a two-column spectrum file, skipping all non-numeric header lines."""
    with open(filepath) as f:
        lines = f.readlines()
    skiprows = 0
    for line in lines:
        try:
            float(line.split()[0])
            break
        except (ValueError, IndexError):
            skiprows += 1
    data = np.loadtxt(filepath, skiprows=skiprows)
    return data[:, 0], data[:, 1]
# ---------------------------------------------------------------------------
# Read background (already normalized)
# ---------------------------------------------------------------------------

def _read_background(source, target, folder=_BKGD_FOLDER):
    fpath = os.path.join(folder, f'{source}-{target}_pdf.txt')
    if not os.path.exists(fpath):
        print(f'measured_spectrum {source} not applicable for {target}')
        return None, None
    print(fpath)
    Er_arr, rate_arr = _read(fpath)
    Er   = Er_arr   * u.keV
    rate = rate_arr * _RATE_UNIT
    if target == 'Ar' and source == 'Rn222':
        rate = rate / 4000
        Er   = Er   * 1000
    return Er, rate


# ---------------------------------------------------------------------------
# Sum sources onto a common Er grid
# ---------------------------------------------------------------------------

def _sum(spectra, Er_common):
    total = np.zeros(len(Er_common))
    for Er, rate in spectra:
        f      = interp1d(Er.to(Er_common.unit).value, rate.value,
                          fill_value=(0, 0), bounds_error=False)
        total += f(Er_common.value)
    return Er_common, total * _RATE_UNIT


# ---------------------------------------------------------------------------
# Smear
# ---------------------------------------------------------------------------

def _smear(Er, rate, target, sigma_percentage=0.07):
    """
    Xe: σ = 0.31√Er + 0.0035·Er  [keV]  (arXiv:1807.07169)
    Ar: σ = sigma_percentage · Er
    """
    Er_keV = Er.to(u.keV)
    if target == 'Xe':
        sigmas = (0.31 * np.sqrt(Er_keV / u.keV) + 0.0035 * Er_keV / u.keV) * u.keV
    elif target == 'Ar':
        sigmas = sigma_percentage * Er_keV
    else:
        raise ValueError(f"Smearing not defined for target '{target}'")

    dEr     = np.diff(Er_keV)
    smeared = np.zeros(len(dEr)) * rate.unit
    for i, (Er_i, dEr_i, sig_i) in enumerate(zip(Er_keV, dEr, sigmas)):
        kernel     = np.exp(-(Er_keV - Er_i)**2 / (2 * sig_i**2)) / (np.sqrt(2 * np.pi) * sig_i)
        smeared[i] = np.sum(rate * kernel * dEr_i)
    return Er_keV[:-1], smeared


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------

def _efficiency(Er_keV, detector, interaction,
                det_eff_folder='../detector_efficiency',
                signal_type='S1S2'):
    if detector == 'LUX03':
        A, B, C, D, E, F = 17.106, 1.8223, 0.65911, 18.292, 20869, -2.35
        return 10 ** (2 - A*np.exp(-B * Er_keV**C)
                        - D*np.exp(-E * Er_keV**F)) / 100

    if 'ideal' in detector and 'Ethrd' in detector:
        Ethrd    = float(detector.split('Ethrd')[-1].replace('keV', '')) * u.keV
        eff_grid = np.logspace(np.log10(Er_keV.value.min()),
                               np.log10(Er_keV.value.max()), 1001) * u.keV
        eff      = np.heaviside(eff_grid - Ethrd, 1.)
        return interp1d(eff_grid, eff, fill_value=(0, 1),
                        bounds_error=False)(Er_keV)

    pattern = os.path.join(det_eff_folder,
                           f'det_eff_{detector}_{interaction}.*')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'No efficiency file matching: {pattern}')
    fpath = files[0]
    if fpath.endswith('.csv'):
        import pandas as pd
        df   = pd.read_csv(fpath, header=0, index_col=0)
        E_   = list(df['E_center [keV]'])
        eff_ = list(df[f'eff {signal_type}'])
    else:
        data  = np.loadtxt(fpath, skiprows=1)
        E_, eff_ = data[:, 0], data[:, 1]
    return interp1d(E_, eff_, fill_value=(0, eff_[-1]),
                    bounds_error=False)(Er_keV)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_detected_spectrum(
        target,
        channel,
        nu_sources,
        bkgd_sources     = None,
        mode             = 'ideal',
        detector         = None,
        metallicity      = 'high',
        Er_range_keV     = None,
        Er_bins          = 501,
        sigma_percentage = 0.1,
        interaction      = None,
        det_eff_folder   = '../detector_efficiency',
        signal_type      = 'S1S2',
        NR_folder        = 'neutrino-Nucleus_el',
        ER_folder        = 'neutrino-electron_el'):
    """
    Parameters
    ----------
    target       : str          'Xe' or 'Ar'
    channel      : str          'NR' or 'ER'
    nu_sources   : list[str]    neutrino sources to include
    bkgd_sources : list[str]    backgrounds to include (optional, ER only)
                                Kr85, Rn222, nubb
    mode         : str          'ideal' or 'realistic'
    detector     : str          required for mode='realistic'
    metallicity  : str          'high' (GS98) or 'low' (AGSS09)

    Returns
    -------
    Er   : astropy Quantity array [keV]
    rate : astropy Quantity array [ton⁻¹ yr⁻¹ keV⁻¹]
    """
    if mode == 'realistic' and detector is None:
        raise ValueError("detector is required for mode='realistic'")

    if Er_range_keV is None:
        Er_range_keV = (0.001, 1000) if channel == 'NR' else (0.001, 5000)
    if interaction is None:
        interaction = 'NR' if channel == 'NR' else 'beta'

    Er_common = np.logspace(np.log10(Er_range_keV[0]),
                            np.log10(Er_range_keV[1]), Er_bins) * u.keV

    spectra = []

    if channel == 'NR':
        for source in nu_sources:
            Er, rate, _ = nr_rate(source, target,
                                  folder=NR_folder, metallicity=metallicity)
            spectra.append((Er, rate))

    elif channel == 'ER':
        for source in nu_sources:
            Er, rate, _ = er_rate(source, target,
                                  folder=ER_folder, metallicity=metallicity)
            spectra.append((Er, rate))
        for source in (bkgd_sources or []):
            Er, rate = _read_background(source, target)
            if Er is not None:
                spectra.append((Er, rate))
    else:
        raise ValueError(f"channel must be 'NR' or 'ER', got '{channel}'")

    Er, rate = _sum(spectra, Er_common)

    if mode == 'realistic':
        Er, rate = _smear(Er, rate, target, sigma_percentage)
        eff      = _efficiency(Er, detector, interaction,
                               det_eff_folder, signal_type)
        rate     = rate * eff

    return Er, rate
