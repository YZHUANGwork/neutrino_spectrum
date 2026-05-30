"""
get_neutrino_continuum_spectrum.py
===================================
High-level entry point for building detected neutrino recoil spectra.
Sits on top of get_elastic_continuum_spectrum.py and wraps the full
pipeline: load pre-computed PDFs → sum sources → smear → apply efficiency.

PIPELINE OVERVIEW
-----------------
The two channels share the same 4-step pipeline:

  1. Load per-source unnormalized PDFs from disk
       neutrino-Nucleus_el/nu-N_<source>-<target>_UNNORMpdf.txt   (NR)
       neutrino-electron_el/nu-e_<source>*UNNORMpdf.txt            (ER)
     These were produced by the _calc scripts.

  2. Apply flux normalization + target number density
       → physical rate dR/dEr  [ton⁻¹ yr⁻¹ keV⁻¹]
     Normalizations live in flux_normalization_uncertainty() inside
     get_elastic_continuum_spectrum.py (solar standard model tables,
     GS98 or AGSS09 metallicity).
     For ER, binding-energy thresholds are also applied per electron shell.

  3. Interpolate all sources onto a common Er grid and sum them
       → get_summed_pdf()

  4. Gaussian smearing  →  gaussian_smearing()
     Xe: σ(Er) = 0.31√Er + 0.0035·Er  (empirical LUX/XENON formula)
     Ar: σ = sigma_percentage · Er     (flat fractional resolution)
     then multiply by detector efficiency  →  get_detector_efficiency()
     Efficiency sources: LUX03 analytic, file lookup, or ideal step function.

PUBLIC API
----------
get_NR_spectrum(nu_sources, target, detectors, ...)
    Returns dict[detector] = {
        'Er_smear'  : energy axis after smearing  [keV]  (astropy Quantity)
        'rate_smear': smeared + efficiency-weighted dR/dEr
        'Er_raw'    : common grid before smearing  [keV]
        'rate_raw'  : summed but un-smeared dR/dEr
    }

get_ER_spectrum(signal_sources, bkgd_sources, target, detectors, ...)
    Same structure but signal and background kept separate:
    Returns dict[detector] = {
        'signal_Er', 'signal_rate',   ← neutrino signal sources
        'bkgd_Er',   'bkgd_rate',     ← Kr85 / Rn222 / nubb backgrounds
        'Er_raw',    'rate_raw',       ← combined, un-smeared
    }

plot_NR_spectrum(result, save_path=None)
plot_ER_spectrum(result, save_path=None)
plot_source_legend(all_sources, save_path=None)  ← requires plot_setup.py

DEPENDENCIES
------------
Required:  numpy, astropy, scipy, matplotlib
           get_elastic_continuum_spectrum.py  (core physics functions)
Optional:  plot_setup.py  (axis formatting + colour/label helpers)
             - setup_cdfpdf_ax : sets axis scales, labels, gridlines, limits
             - get_colors      : consistent source colour map
             - get_official_pcle : LaTeX particle labels (e.g. ⁷Be, ¹³N)
           If plot_setup.py is absent the plot functions still work with
           plain matplotlib defaults.

FILE FORMAT (txt spectra on disk)
----------------------------------
  Er [keV]\t\tdN_dEr [cm2 / keV]
  1.0000E-03\t\t2.5542E-39
  ...
  Tab-separated, one header line, two float columns.
  Read natively with:  data = np.loadtxt(file, skiprows=1)
  No dependency on read.py needed.

Converted from: plot_det_neutrino_continuum_0331.ipynb
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.patches import Patch

# plot_setup.py is purely cosmetic (axis formatting, colours, LaTeX labels).
# Import it when present; all spectrum functions work without it.
try:
    from plot_setup import setup_cdfpdf_ax, get_colors, get_official_pcle
    _HAS_PLOT_SETUP = True
except ImportError:
    _HAS_PLOT_SETUP = False

from get_elastic_continuum_spectrum import (
    get_neutrino_Nucleus_el_pdf,
    get_neutrino_electron_el_pdf,
    get_detected_pdf,
    get_summed_pdf,
)

# ---------------------------------------------------------------------------
# Default units
# ---------------------------------------------------------------------------
_dxsec = u.cm**2 / u.keV
_dRdEr = 1 / u.tonne / u.yr / u.keV


# ---------------------------------------------------------------------------
# NR (neutrino-nucleus) spectrum
# ---------------------------------------------------------------------------

def get_NR_spectrum(
    nu_sources,
    target           = 'Xe',
    detectors        = ('LZ',),
    NR_folder        = '../neutrino_spectrum/neutrino-Nucleus_el',
    metallicity      = 'high',
    Er_range_keV     = (0.001, 1000),
    Er_bins          = 501,
    sigma_percentage = 0.1,
    interaction      = 'NR',
):
    """
    Load NR PDFs, sum over sources, apply detector smearing + efficiency.

    Parameters
    ----------
    nu_sources : list[str]
        e.g. ['pp', 'Be7_384', 'Be7_861', 'pep', 'N13', 'O15', 'F17',
               '8B', 'hep', 'dsnb', 'atmNu_SURF_avg']
    target : str
        'Xe' or 'Ar'
    detectors : list[str]
        Detector names understood by get_detector_efficiency().
    NR_folder : str
        Path to the neutrino-Nucleus_el pre-computed PDF folder.
    metallicity : str
        'high' (GS98) or 'low' (AGSS09).
    Er_range_keV : (float, float)
        Log-spaced recoil energy grid limits [keV].
    Er_bins : int
        Number of points in the common Er grid.
    sigma_percentage : float
        Fractional energy resolution (used for Ar; Xe uses built-in formula).
    interaction : str
        Passed to get_detector_efficiency(); typically 'NR'.

    Returns
    -------
    dict  { detector_name: { 'Er_smear', 'rate_smear', 'Er_raw', 'rate_raw' } }
    """
    source_dict = get_neutrino_Nucleus_el_pdf(
        nu_sources, target,
        folder         = NR_folder,
        metallicity    = metallicity,
        dxsec_dEr_unit = _dxsec,
        dR_dEr_unit    = _dRdEr,
        plot           = False,
    )

    Er_common = np.logspace(
        np.log10(Er_range_keV[0]), np.log10(Er_range_keV[1]), Er_bins
    ) * u.keV

    result = {}
    for detector in detectors:
        Er_smear, rate_smear, Er_raw, rate_raw = get_detected_pdf(
            Er_common, source_dict, target, detector,
            interaction, sigma_percentage=sigma_percentage,
        )
        result[detector] = {
            'Er_smear'  : Er_smear,
            'rate_smear': rate_smear,
            'Er_raw'    : Er_raw,
            'rate_raw'  : rate_raw,
        }
    return result


# ---------------------------------------------------------------------------
# ER (neutrino-electron) spectrum
# ---------------------------------------------------------------------------

def get_ER_spectrum(
    signal_sources,
    bkgd_sources,
    target           = 'Ar',
    detectors        = ('ideal Ethrd1keV',),
    ER_folder        = '../neutrino_spectrum/neutrino-electron_el',
    metallicity      = 'high',
    Er_range_keV     = (0.001, 5000),
    Er_bins          = 501,
    sigma_percentage = 0.1,
    interaction      = 'beta',
):
    """
    Load ER PDFs for signal and background sources separately,
    apply detector smearing + efficiency.

    Parameters
    ----------
    signal_sources : list[str]
        e.g. ['pp', 'Be7_384', 'Be7_861', 'pep', 'N13', 'O15', 'F17']
    bkgd_sources : list[str]
        e.g. ['nubb', 'Kr85', 'Rn222']
    (other parameters identical to get_NR_spectrum)

    Returns
    -------
    dict  { detector_name: { 'signal_Er', 'signal_rate',
                              'bkgd_Er',   'bkgd_rate',
                              'Er_raw',    'rate_raw' } }
    """
    def _load(sources):
        return get_neutrino_electron_el_pdf(
            sources, target,
            folder         = ER_folder,
            metallicity    = metallicity,
            dxsec_dEr_unit = _dxsec,
            dR_dEr_unit    = _dRdEr,
            plot           = False,
        )

    signal_dict = _load(signal_sources)
    bkgd_dict   = _load(bkgd_sources)

    Er_common = np.logspace(
        np.log10(Er_range_keV[0]), np.log10(Er_range_keV[1]), Er_bins
    ) * u.keV

    result = {}
    for detector in detectors:
        sig_Er, sig_rate, Er_raw, rate_raw = get_detected_pdf(
            Er_common, signal_dict, target, detector,
            interaction, sigma_percentage=sigma_percentage,
        )
        bkgd_Er, bkgd_rate, _, _ = get_detected_pdf(
            Er_common, bkgd_dict, target, detector,
            interaction, sigma_percentage=sigma_percentage,
        )
        result[detector] = {
            'signal_Er'  : sig_Er,
            'signal_rate': sig_rate,
            'bkgd_Er'    : bkgd_Er,
            'bkgd_rate'  : bkgd_rate,
            'Er_raw'     : Er_raw,
            'rate_raw'   : rate_raw,
        }
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _apply_ax_style(ax, xlabel, xlims, ylims):
    """Minimal axis styling used when plot_setup.py is unavailable."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$d\mathcal{R}/dE_r$  [ton$^{-1}$ yr$^{-1}$ keV$^{-1}$]')
    ax.set_xscale('log');  ax.set_yscale('log')
    ax.set_xlim(*xlims);   ax.set_ylim(*ylims)
    ax.grid(True)


def plot_NR_spectrum(
    result,
    save_path = None,
    xlims     = (0.001, 90),
    ylims     = (1e-5, 1e7),
):
    """
    Plot smeared NR spectra for each detector on a single axes.

    Parameters
    ----------
    result : dict
        Output of get_NR_spectrum().
    save_path : str or None
        If given, save the figure to this path.

    Returns
    -------
    fig, ax
    """
    lines = ['--', ':', '-.', '-', (0, (3, 1, 1, 1))]
    fig, ax = plt.subplots(figsize=(6, 4))

    for d, (detector, vals) in enumerate(result.items()):
        ax.loglog(
            vals['Er_smear'], vals['rate_smear'],
            color='black', ls=lines[d % len(lines)], label=detector,
        )

    if _HAS_PLOT_SETUP:
        ax = setup_cdfpdf_ax(
            ax, '', 'Nucleus recoil kinetic energy Er [keV]', '', True, 'pdf', 15, 15,
            vlines=[], hlines=[], xlims=list(xlims), ylims=list(ylims), log=[0, 0],
        )
    else:
        _apply_ax_style(ax, 'Nucleus recoil kinetic energy Er [keV]', xlims, ylims)

    ax.legend(fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig, ax


def plot_ER_spectrum(
    result,
    save_path = None,
    xlims     = (1e-3, 3000),
    ylims     = (1e-5, 200),
):
    """
    Plot smeared ER spectra for each detector.
    Signal sources are drawn in black; background sources in grey.

    Parameters
    ----------
    result : dict
        Output of get_ER_spectrum().
    save_path : str or None
        If given, save the figure to this path.

    Returns
    -------
    fig, ax
    """
    lines = ['--', ':', '-.', '-', (0, (3, 1, 1, 1))]
    fig, ax = plt.subplots(figsize=(6, 4))

    for d, (detector, vals) in enumerate(result.items()):
        ls    = lines[d % len(lines)]
        label = detector.split(' ')[-1]
        ax.loglog(vals['bkgd_Er'],   vals['bkgd_rate'],  color='grey',  ls=ls)
        ax.loglog(vals['signal_Er'], vals['signal_rate'], color='black', ls=ls, label=label)

    if _HAS_PLOT_SETUP:
        ax = setup_cdfpdf_ax(
            ax, '', 'Electron recoil kinetic energy Er [keV]', '', True, 'pdf', 15, 15,
            vlines=[], hlines=[], xlims=list(xlims), ylims=list(ylims), log=[0, 0],
        )
    else:
        _apply_ax_style(ax, 'Electron recoil kinetic energy Er [keV]', xlims, ylims)

    ax.legend(fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig, ax


def plot_source_legend(all_sources, save_path=None):
    """
    Standalone colour-coded legend panel for all neutrino sources.
    Requires plot_setup.py (get_colors, get_official_pcle).
    """
    if not _HAS_PLOT_SETUP:
        raise ImportError(
            "plot_source_legend requires plot_setup.py "
            "(get_colors and get_official_pcle must be available)"
        )
    legend_elements = [
        Patch(facecolor=get_colors(src)[0], label=get_official_pcle(src))
        for src in all_sources
    ]
    fig, ax = plt.subplots()
    ax.axis(False)
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.0, 0.0, 2, 1), loc='lower left',
        ncol=5, mode='expand', borderaxespad=0., fontsize=15,
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# CLI / demo  (python get_neutrino_continuum_spectrum.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # ---- NR example (Xe target, multiple detectors) ----------------------
    NR_sources   = ['pp', 'Be7_384', 'Be7_861', 'pep',
                    'N13', 'O15', 'F17', '8B', 'hep',
                    'dsnb', 'atmNu_SURF_avg']
    NR_detectors = ['Xe100t-5', 'G3', 'LZ', 'Xe1t']

    nr_result = get_NR_spectrum(
        nu_sources  = NR_sources,
        target      = 'Xe',
        detectors   = NR_detectors,
        interaction = 'NR',
    )
    # Access arrays: nr_result['LZ']['Er_smear'], nr_result['LZ']['rate_smear']
    plot_NR_spectrum(nr_result, save_path=os.path.join('FIGURE', 'pdf_Xe_NR_dets.pdf'))

    # ---- ER example (Ar target, ideal threshold detectors) ---------------
    signal_sources = ['pp', 'Be7_384', 'Be7_861', 'pep', 'N13', 'O15', 'F17']
    bkgd_sources   = ['nubb', 'Kr85', 'Rn222']
    ER_detectors   = ['ideal Ethrd100keV', 'ideal Ethrd40keV', 'ideal Ethrd1keV']

    er_result = get_ER_spectrum(
        signal_sources = signal_sources,
        bkgd_sources   = bkgd_sources,
        target         = 'Ar',
        detectors      = ER_detectors,
        interaction    = 'beta',
    )
    # Access arrays: er_result['ideal Ethrd1keV']['signal_Er'], ['signal_rate']
    plot_ER_spectrum(er_result, save_path=os.path.join('FIGURE', 'pdf_Ar_beta_dets.pdf'))

    # ---- Legend panel (needs plot_setup.py) ------------------------------
    all_sources = ['pp', 'Be7_384', 'Be7_861', 'pep', 'N13', 'O15', 'F17',
                   '8B', 'hep', 'dsnb', 'atmNu_SURF_avg', 'nubb', 'Kr85', 'Rn222']
    plot_source_legend(all_sources, save_path=os.path.join('FIGURE', 'all_sources.pdf'))
