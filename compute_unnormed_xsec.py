"""
compute_unnormed_xsec.py
=========================
Computes unnormalized cross-section spectra dσ/dEr [cm²/keV] for a given
neutrino source and detector target, and saves them to disk.

This is the first step in the pipeline. The output files are later read by
cross_section_to_rate.py which multiplies by flux normalization and target
number density to get a physical event rate [ton⁻¹ yr⁻¹ keV⁻¹].

NR (neutrino-nucleus elastic)
------------------------------
Output: neutrino-Nucleus_el/nu-N_<source>-<target>_UNNORMpdf.txt
Sources: pp, Be7_384, Be7_861, pep, N13, O15, F17, 8B, hep, atmNu_SURF_avg

ER (neutrino-electron elastic)
--------------------------------
Output: neutrino-electron_el/nu-e_<source>-<flavor>_UNNORMpdf.txt
        one file per flavor: nue (CCNC) and numutau (NC)
Sources: pp, Be7_384, Be7_861, pep, N13, O15, F17

What this script computes
--------------------------
For a continuum neutrino source (e.g. pp, 8B, atmNu):
    dσ/dEr(Er) = ∫ dσ/dEr(Er, Eν) × dΦ/dEν × Pee(Eν)  dEν   [cm²/keV]

For a mono-energetic source (e.g. Be7_861, pep):
    dσ/dEr(Er) = dσ/dEr(Er, Eν) × Pee(Eν)                    [cm²/keV]

Pee is the energy-dependent electron neutrino survival probability.
    nue flavor:     weight by Pee(Eν)
    numutau flavor: weight by 1 - Pee(Eν)

The flux shape dΦ/dEν comes from neutrino_flux.py.
The differential cross sections come from scatter_xsec_el.py.

Usage
-----
    python compute_unnormed_xsec.py

Edit the configuration block at the bottom to change source/target.
Or import and call compute_NR_unnormed() / compute_ER_unnormed() directly.
"""

import numpy as np
import os
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt

from scatter_kinematics import ermax_el, ermin_el
from scatter_target import get_target_info
from scatter_xsec_el import delxsec_dEr_nucleus, delxsec_dEr_lepton
from neutrino_flux import flux_spectrum


def get_Edep_Pee(Enus_MeV, pee_file="Pee.txt"):
    """
    Energy-dependent electron neutrino survival probability Pee(Ev).
    Interpolated from Pee.txt.  ref: https://arxiv.org/pdf/1208.5723.pdf

    Pee.txt: two columns -- E_nu [MeV], Pee  (first line is URL comment)

    Parameters
    ----------
    Enus_MeV : astropy Quantity array [MeV]
    pee_file : str   path to Pee.txt

    Returns
    -------
    Pee values interpolated at Enus_MeV (dimensionless)
    """
    data     = np.loadtxt(pee_file, skiprows=1)
    E_nu_Pee = data[:, 0] * u.MeV
    Pee      = data[:, 1]
    return np.interp(Enus_MeV, E_nu_Pee, Pee)


_XSEC_UNIT = u.cm**2 / u.keV
_NR_FOLDER = 'neutrino-Nucleus_el'
_ER_FOLDER = 'neutrino-electron_el'


def _save(outfile, Ers, dσ_dEr):
    header = f'Er [{Ers.unit}]\t\tdN_dEr [{dσ_dEr.unit}]'
    np.savetxt(outfile,
               np.column_stack([Ers.value, dσ_dEr.value]),
               header=header, comments='', fmt='%.10E', delimiter='\t\t')
    print(f'Saved: {outfile}')


def _check_plot(Ers, dσ_dEr, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(Ers, dσ_dEr)
    ax.set_xlabel('Er [keV]')
    ax.set_ylabel(f'dσ/dEr [{_XSEC_UNIT}]')
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# NR: neutrino-nucleus elastic
# ---------------------------------------------------------------------------

def compute_NR_unnormed(source, target,
                        Er_bins   = 501,
                        folder    = _NR_FOLDER,
                        plot      = False,
                        overwrite = True):
    """
    Compute and save dσ/dEr [cm²/keV] for one NR neutrino source and target.

    Parameters
    ----------
    source    : str    e.g. 'pp', 'Be7_861', '8B', 'atmNu_SURF_avg'
    target    : str    'Xe' or 'Ar'
    Er_bins   : int    number of recoil energy grid points
    folder    : str    output directory
    plot      : bool   plot result to check (does not save figure)
    overwrite : bool   skip if output file already exists

    Returns
    -------
    Er    : astropy Quantity array [keV]
    dσ_dEr: astropy Quantity array [cm²/keV]
    """
    A, Z = get_target_info(target)
    m_N  = A * (const.m_n * const.c**2).to(u.GeV)

    # neutrino flux shape
    E_neus, dN_dEneus = flux_spectrum(source, E_neu_unit=u.MeV)

    continuum = not np.isscalar(dN_dEneus.value)
    if continuum:
        E_neu_centers         = (E_neus[1:] + E_neus[:-1]) / 2
        dN_dEneus_Eneucenters = (dN_dEneus[1:] + dN_dEneus[:-1]) / 2
        dE_neus               = np.diff(E_neus)
        Er_max = ermax_el(max(E_neus), m_N).to(u.keV)
        Er_min = ermin_el(min(E_neus), m_N).to(u.keV)
    else:
        Er_max = ermax_el(E_neus, m_N).to(u.keV)
        Er_min = ermin_el(E_neus, m_N).to(u.keV)

    Ers = np.logspace(-3, np.log10(np.ceil(Er_max.value * 2000) / 2000),
                      Er_bins) * u.keV

    os.makedirs(folder, exist_ok=True)
    outfile = os.path.join(folder, f'nu-N_{source}-{target}_UNNORMpdf.txt')
    if os.path.exists(outfile) and not overwrite:
        print(f'[skip] {outfile}')
        data = np.loadtxt(outfile, skiprows=1)
        return data[:, 0] * u.keV, data[:, 1] * _XSEC_UNIT

    dN_dErs = np.ones(len(Ers))
    for er, Er in enumerate(Ers):
        if continuum:
            delxsec_dEr_Eneucenters = np.ones(len(E_neu_centers))
            for e, E_neu_center in enumerate(E_neu_centers):
                delxsec_dEr_Eneucenters[e] = delxsec_dEr_nucleus(
                    E_neu_center, Er, A, Z, unit_dxecdEr=_XSEC_UNIT
                ).dxsec_dEr().value
            delxsec_dEr_Eneucenters *= _XSEC_UNIT
            dN_dErs[er] = sum(
                delxsec_dEr_Eneucenters * dN_dEneus_Eneucenters * dE_neus
            ).to(_XSEC_UNIT).value
        else:
            dN_dErs[er] = delxsec_dEr_nucleus(
                E_neus, Er, A, Z, unit_dxecdEr=_XSEC_UNIT
            ).dxsec_dEr().value

    dN_dErs *= _XSEC_UNIT

    _save(outfile, Ers, dN_dErs)
    if plot:
        _check_plot(Ers, dN_dErs, f'NR  {source} — {target}')
    return Ers, dN_dErs


# ---------------------------------------------------------------------------
# ER: neutrino-electron elastic
# ---------------------------------------------------------------------------

def compute_ER_unnormed(source, target,
                        flavor    = 'both',
                        Er_bins   = 501,
                        folder    = _ER_FOLDER,
                        plot      = False,
                        overwrite = True):
    """
    Compute and save dσ/dEr [cm²/keV] for one ER neutrino source and target.
    Saves one file per neutrino flavor.

    Parameters
    ----------
    source    : str    e.g. 'pp', 'Be7_861', 'pep', 'N13', 'O15', 'F17'
    target    : str    'Xe' or 'Ar'  (used only for the Er grid upper limit)
    flavor    : str    'nue', 'numutau', or 'both' (default)
    Er_bins   : int    number of recoil energy grid points
    folder    : str    output directory
    plot      : bool   plot result to check (does not save figure)
    overwrite : bool   skip if output file already exists

    Returns
    -------
    dict { flavor: (Er, dσ_dEr) }
    """
    m_e = (const.m_e * const.c**2).to(u.keV)

    # neutrino flux shape
    E_neus, dP_dEneus = flux_spectrum(source, E_neu_unit=u.MeV)

    continuum = not np.isscalar(dP_dEneus.value)
    if continuum:
        E_neu_centers          = (E_neus[1:] + E_neus[:-1]) / 2
        dP_dEneus_Eneucenters  = (dP_dEneus[1:] + dP_dEneus[:-1]) / 2
        dE_neus                = np.diff(E_neus)
        Er_max = ermax_el(max(E_neus), m_e).to(u.keV)
        Er_min = ermin_el(min(E_neus), m_e).to(u.keV)
    else:
        Er_max = ermax_el(E_neus, m_e).to(u.keV)
        Er_min = ermin_el(E_neus, m_e).to(u.keV)

    Ers = np.logspace(-3, np.log10(np.ceil(Er_max.value * 2000) / 2000),
                      Er_bins) * u.keV

    os.makedirs(folder, exist_ok=True)
    flavors = ['nue', 'numutau'] if flavor == 'both' else [flavor]
    results = {}

    for neutrino_flavor in flavors:
        if neutrino_flavor == 'nue':
            current = 'CCNC'
            Pee_E = get_Edep_Pee(E_neu_centers) if continuum \
                    else get_Edep_Pee(E_neus)
        elif neutrino_flavor == 'numutau':
            current = 'NC'
            Pee_E = 1 - get_Edep_Pee(E_neu_centers) if continuum \
                    else 1 - get_Edep_Pee(E_neus)

        outfile = os.path.join(folder,
                               f'nu-e_{source}-{current}{neutrino_flavor}_UNNORMpdf.txt')
        if os.path.exists(outfile) and not overwrite:
            print(f'[skip] {outfile}')
            data = np.loadtxt(outfile, skiprows=1)
            results[neutrino_flavor] = (data[:, 0] * u.keV, data[:, 1] * _XSEC_UNIT)
            continue

        dN_dErs = np.ones(len(Ers))
        for er, Er in enumerate(Ers):
            if continuum:
                delxsec_dEr_Eneucenters = np.ones(len(E_neu_centers))
                for e, E_neu_center in enumerate(E_neu_centers):
                    delxsec_dEr_Eneucenters[e] = delxsec_dEr_lepton(
                        E_neu_center, Er, neutrino_flavor, unit_dxecdEr=_XSEC_UNIT
                    ).dxsec_dEr().value
                delxsec_dEr_Eneucenters *= _XSEC_UNIT
                dN_dEr = sum(
                    delxsec_dEr_Eneucenters * dP_dEneus_Eneucenters * Pee_E * dE_neus
                )
                dN_dErs[er] = dN_dEr.to(_XSEC_UNIT).value
            else:
                delxsec_dEr_E_neu = delxsec_dEr_lepton(
                    E_neus, Er, neutrino_flavor, unit_dxecdEr=_XSEC_UNIT
                ).dxsec_dEr()
                dN_dEr = delxsec_dEr_E_neu * dP_dEneus * Pee_E * E_neus.unit
                dN_dErs[er] = dN_dEr.to(_XSEC_UNIT).value

        dN_dErs *= _XSEC_UNIT

        _save(outfile, Ers, dN_dErs)
        if plot:
            _check_plot(Ers, dN_dErs, f'ER  {source} {neutrino_flavor} — {target}')
        results[neutrino_flavor] = (Ers, dN_dErs)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # NR
    NR_TARGET  = 'Xe'
    NR_SOURCES = ['pp', 'Be7_384', 'Be7_861', 'pep',
                  'N13', 'O15', 'F17', '8B', 'hep', 'atmNu_SURF_avg']
    for source in NR_SOURCES:
        compute_NR_unnormed(source, NR_TARGET, plot=False,overwrite = True)

    # ER
    ER_TARGET  = 'Xe'
    ER_SOURCES = ['pp', 'Be7_384', 'Be7_861', 'pep', 'N13', 'O15', 'F17']
    for source in ER_SOURCES:
        compute_ER_unnormed(source, ER_TARGET, flavor='both', plot=False,overwrite = True)
