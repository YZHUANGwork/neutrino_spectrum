"""
neutrino_flux.py
=================
Neutrino flux shapes and normalizations.

flux_spectrum(source)
    Returns the flux shape (normalized to 1) for a given source.
    Continuum sources: read from local txt files in solar_neutrino_flux/ or atm_neutrino_flux/
    Mono-energetic sources: return a single energy value.

flux_normalization_uncertainty(source, metallicity)
    Returns (normalization, fractional_uncertainty).
    ref: https://arxiv.org/pdf/1208.5723.pdf  (normalizations)
         https://arxiv.org/pdf/1307.5458      (uncertainties)
    metallicity 'high' = GS98-SFII,  'low' = AGSS09-SFII

Sources
-------
Continuum solar : pp, hep, 8B, N13, O15, F17
Mono-energetic  : Be7_384, Be7_861, pep
Atmospheric     : atmNu_<site>_<year>  e.g. atmNu_SURF_avg
"""

import numpy as np
import os
import astropy.units as u

_SOLAR_FOLDER = 'solar_neutrino_flux'
_ATM_FOLDER   = 'atm_neutrino_flux'


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
# Flux shape (normalized to 1)
# ---------------------------------------------------------------------------

def flux_spectrum(source, E_neu_unit=u.MeV):
    """
    Return the neutrino flux energy spectrum normalized to 1.

    Parameters
    ----------
    source     : str
    E_neu_unit : astropy Unit   energy unit for returned arrays

    Returns
    -------
    E_neu    : astropy Quantity   neutrino energy (scalar for mono-energetic)
    dN_dEneu : astropy Quantity   flux shape [1/E_neu_unit]
                                  (scalar 1/E_neu_unit for mono-energetic)
    """
    # --- continuum solar ---
    if source in ['pp', 'hep', '8B', 'N13', 'O15', 'F17']:
        fpath        = os.path.join(_SOLAR_FOLDER, f'{source}.txt')
        E_neu, dN    = _read(fpath)
        return E_neu * E_neu_unit, dN / E_neu_unit

    # --- mono-energetic solar ---
    elif source in ['Be7_384', 'Be7_861', 'pep']:
        energies = {'Be7_384': 0.38447, 'Be7_861': 0.86227, 'pep': 1.442}
        return energies[source] * E_neu_unit, 1 / E_neu_unit

    # --- atmospheric ---
    elif source.startswith('atm'):
        site = source.split('_')[1]
        year = source.split('_')[-1]

        if 'FLUKA' in year:
            fpath       = os.path.join(_ATM_FOLDER, f'{site}_{year}_perGeVm2s.txt')
            E_GeV, dN   = _read(fpath)
            dN_unit     = (dN / (u.GeV * u.m**2 * u.s)).to(1 / E_neu_unit / u.cm**2 / u.s)
        else:
            fpath       = os.path.join(_ATM_FOLDER, site,
                                       f'{site}_{year}_Trackback_perGeVcm2s.txt')
            E_GeV, dN   = _read(fpath)
            dN_unit     = (dN / (u.GeV * u.cm**2 * u.s)).to(1 / E_neu_unit / u.cm**2 / u.s)

        E_neu        = (E_GeV * u.GeV).to(E_neu_unit)
        norm         = np.trapz(dN_unit[:-1].value, np.diff(E_neu.value)) * dN_unit.unit * E_neu_unit
        dN_dEneu     = (dN_unit / norm).to(1 / E_neu_unit)
        return E_neu, dN_dEneu

    else:
        raise ValueError(f"Unknown neutrino source: '{source}'")


# ---------------------------------------------------------------------------
# Flux normalization
# ---------------------------------------------------------------------------

def flux_normalization_uncertainty(source, metallicity='high',
                                   norm_unit=(u.cm**2 * u.s)**(-1)):
    """
    Return (normalization, fractional_uncertainty) for a neutrino source.

    Parameters
    ----------
    source      : str
    metallicity : str   'high' (GS98) or 'low' (AGSS09)
    norm_unit   : astropy Unit

    Returns
    -------
    normalization : astropy Quantity
    sigma         : float   fractional 1-sigma uncertainty
    """
    hi = (metallicity == 'high')

    table = {
        'pp':      (5.98e10 if hi else 6.03e10,  0.01),
        'pep':     (1.44e8  if hi else 1.47e8,   0.),
        'hep':     (8.04e3  if hi else 8.31e3,   0.),
        '8B':      (5.58e6  if hi else 4.59e6,   0.16),
        'Be7_384': ((5.00e9 if hi else 4.56e9) * 0.1, 0.105),
        'Be7_861': ((5.00e9 if hi else 4.56e9) * 0.9, 0.105),
        'N13':     (2.96e8  if hi else 2.17e8,   0.3),
        'O15':     (2.23e8  if hi else 1.56e8,   0.3),
        'F17':     (5.52e6  if hi else 3.40e6,   0.3),
    }

    if source in table:
        val, sigma = table[source]
        return val * norm_unit, sigma

    if source.startswith('atm'):
        sigma = 0.2
        site  = source.split('_')[1]
        year  = source.split('_')[-1]

        if 'FLUKA' in year:
            fpath     = os.path.join(_ATM_FOLDER, f'{site}_{year}_perGeVm2s.txt')
            E_GeV, dN = _read(fpath)
            dN_unit   = dN / (u.GeV * u.m**2 * u.s)
        else:
            fpath     = os.path.join(_ATM_FOLDER, site,
                                     f'{site}_{year}_Trackback_perGeVcm2s.txt')
            E_GeV, dN = _read(fpath)
            dN_unit   = dN / (u.GeV * u.cm**2 * u.s)

        E_GeV    *= u.GeV
        norm      = np.trapz(dN_unit[:-1].value, np.diff(E_GeV.value)) * dN_unit.unit * u.GeV
        return norm.to(norm_unit), sigma

    raise ValueError(f"Unknown neutrino source: '{source}'")
