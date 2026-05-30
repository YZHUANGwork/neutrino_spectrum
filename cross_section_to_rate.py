"""
cross_section_to_rate.py
=========================
Converts pre-computed unnormalized cross-section spectra [cm²/keV]
to physical event rate spectra [ton⁻¹ yr⁻¹ keV⁻¹] for a given
neutrino source and detector target.

    dR/dEr [ton⁻¹ yr⁻¹ keV⁻¹]
        = dσ/dEr [cm²/keV]
        × flux_norm [cm⁻² s⁻¹]
        × N_target [ton⁻¹]

One source + one target per call.

NR (neutrino-nucleus):   pp, Be7_384, Be7_861, pep, N13, O15, F17,
                         8B, hep, dsnb, atmNu_SURF_avg
ER (neutrino-electron):  pp, Be7_384, Be7_861, pep, N13, O15, F17

Flux normalizations
-------------------
GS98   (metallicity='high'): arXiv:1208.5723
AGSS09 (metallicity='low'):  arXiv:1307.5458
"""

import numpy as np
import os
import glob
import itertools
import astropy.units as u
import astropy.constants as const

_CM2_PER_KEV = u.cm**2 / u.keV
_RATE_UNIT   = 1 / u.tonne / u.yr / u.keV


# ---------------------------------------------------------------------------
# Flux normalization table
# ---------------------------------------------------------------------------

def _flux_norm(source, metallicity):
    hi  = (metallicity == 'high')
    tbl = {
        'pp':      (5.98e10 if hi else 6.03e10,  0.01),
        'pep':     (1.44e8  if hi else 1.47e8,   0.017),
        'hep':     (8.04e3  if hi else 8.31e3,   0.155),
        '8B':      (5.58e6  if hi else 4.59e6,   0.16),
        'Be7_384': ((5.00e9 if hi else 4.56e9) * 0.1, 0.105),
        'Be7_861': ((5.00e9 if hi else 4.56e9) * 0.9, 0.105),
        'N13':     (2.96e8  if hi else 2.17e8,   0.3),
        'O15':     (2.23e8  if hi else 1.56e8,   0.3),
        'F17':     (5.52e6  if hi else 3.40e6,   0.3),
        'dsnb':    (0.,                           0.5),
    }
    if source in tbl:
        val, sigma = tbl[source]
        return val * (u.cm**2 * u.s)**(-1), sigma
    if source.startswith('atm'):
        return _atm_flux_norm(source)
    raise ValueError(f"Unknown neutrino source: '{source}'")


def _atm_flux_norm(source):
    ATM_FOLDER = './atm_neutrino_flux'
    sigma = 0.2
    site  = source.split('_')[1]
    year  = source.split('_')[-1]
    if 'FLUKA' in year:
        fpath = os.path.join(ATM_FOLDER, f'{site}_{year}_perGeVm2s.txt')
        data  = np.loadtxt(fpath, skiprows=1)
        E, dN = data[:, 0] * u.GeV, data[:, 1] / (u.GeV * u.m**2 * u.s)
    else:
        fpath = os.path.join(ATM_FOLDER, site,
                             f'{site}_{year}_Trackback_perGeVcm2s.txt')
        data  = np.loadtxt(fpath, skiprows=1)
        E, dN = data[:, 0] * u.GeV, data[:, 1] / (u.GeV * u.cm**2 * u.s)
    norm = np.trapz(dN[:-1].value, np.diff(E.value)) * dN.unit * E.unit
    return norm.to((u.cm**2 * u.s)**(-1)), sigma


# ---------------------------------------------------------------------------
# Target properties
# ---------------------------------------------------------------------------

def _target_number_density(target):
    tbl = {'Xe': 131.293, 'Ar': 39.948, 'He': 4.}
    if target not in tbl:
        raise ValueError(f"Unknown target '{target}'. Available: {list(tbl)}")
    return const.N_A / (1 * u.g / u.mol) / tbl[target]


def _electron_binding_energies(target):
    shells = {
        'Xe': ([34563]*2   + [5454.8]*2  + [4891.4]*6  + [1148]*2   +
               [959.78]*6  + [681.02]*10 + [214.63]*2  + [153.48]*6 +
               [68.146]*10 + [23.39]*2   + [12.563]*6),
        'Ar': ([3206.2]*2  + [324.2]*2   + [247.74]*6  +
               [29.24]*2   + [15.76]*6),
    }
    if target not in shells:
        raise ValueError(f"Binding energies not defined for '{target}'")
    return np.array(shells[target]) * u.eV


# ---------------------------------------------------------------------------
# NR: neutrino-nucleus elastic
# ---------------------------------------------------------------------------

def nr_rate(source, target,
            folder      = 'neutrino-Nucleus_el',
            metallicity = 'high'):
    """
    Convert unnormed NR cross-section spectrum to event rate.

    Parameters
    ----------
    source : str    e.g. 'pp', 'Be7_861', '8B', 'atmNu_SURF_avg', 'dsnb'
    target : str    'Xe' or 'Ar'
    folder : str    path to neutrino-Nucleus_el directory
    metallicity : str   'high' (GS98) or 'low' (AGSS09)

    Returns
    -------
    Er          : astropy Quantity array [keV]
    rate        : astropy Quantity array [ton⁻¹ yr⁻¹ keV⁻¹]
    uncertainty : float   fractional 1-sigma flux uncertainty
    """
    N_target = _target_number_density(target)

    if source == 'dsnb':
        fpath = os.path.join(folder, f'nu-N_{source}-{target}_pdf.txt')
        data  = np.loadtxt(fpath, skiprows=1)
        return data[:, 0] * u.keV, data[:, 1] * _RATE_UNIT, 0.5

    fpath = os.path.join(folder, f'nu-N_{source}-{target}_UNNORMpdf.txt')
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)

    norm, uncertainty = _flux_norm(source, metallicity)
    data = np.loadtxt(fpath, skiprows=1)
    Er   = data[:, 0] * u.keV
    rate = (N_target * norm * (data[:, 1] * _CM2_PER_KEV)).to(_RATE_UNIT)
    return Er, rate, uncertainty


# ---------------------------------------------------------------------------
# ER: neutrino-electron elastic
# ---------------------------------------------------------------------------

def er_rate(source, target,
            folder      = 'neutrino-electron_el',
            metallicity = 'high'):
    """
    Convert unnormed ER cross-section spectrum to event rate.
    Sums over neutrino flavors (nue + numutau) and applies per-shell
    electron binding energy thresholds.

    Parameters
    ----------
    source : str    e.g. 'pp', 'Be7_861', 'pep', 'N13', 'O15', 'F17'
    target : str    'Xe' or 'Ar'
    folder : str    path to neutrino-electron_el directory
    metallicity : str   'high' (GS98) or 'low' (AGSS09)

    Returns
    -------
    Er          : astropy Quantity array [keV]
    rate        : astropy Quantity array [ton⁻¹ yr⁻¹ keV⁻¹]
    uncertainty : float   fractional 1-sigma flux uncertainty
    """
    N_target = _target_number_density(target)
    E_bind   = _electron_binding_energies(target)

    files = glob.glob(os.path.join(folder, f'nu-e_{source}*UNNORMpdf.txt'))
    if not files:
        raise FileNotFoundError(
            f"No ER files found for source='{source}' in {folder}")

    norm, uncertainty = _flux_norm(source, metallicity)

    all_Er, all_dσ = [], []
    for f in files:
        data = np.loadtxt(f, skiprows=1)
        all_Er.append(list(data[:, 0]))
        all_dσ.append(data[:, 1] * _CM2_PER_KEV)

    all_Er.sort()
    unique = list(g for g, _ in itertools.groupby(all_Er))
    if len(unique) != 1:
        raise ValueError(f"Er grids differ across flavor files for '{source}'")

    Er_keV   = np.array(unique[0]) * u.keV
    dσ_total = sum(all_dσ)
    dσ_electrons = sum((Er_keV > E) * dσ_total for E in E_bind)

    rate = (N_target * norm * dσ_electrons).to(_RATE_UNIT)
    return Er_keV, rate, uncertainty
