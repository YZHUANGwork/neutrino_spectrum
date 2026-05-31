import numpy as np
import os

import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.patches import Patch

from plot_setup import setup_cdfpdf_ax, get_colors, get_official_pcle
from cross_section_to_rate import nr_rate, er_rate
from get_detected_spectrum import get_detected_spectrum, _read_background

os.makedirs('figures', exist_ok=True)

NR_SOURCES_XE = ['pp', 'Be7_384', 'Be7_861', 'pep',
                  'N13', 'O15', 'F17', '8B', 'hep',
                  'dsnb', 'atmNu_SURF_avg']
NR_SOURCES_AR = ['pp', 'Be7_384', 'Be7_861', 'pep',
                  'N13', 'O15', 'F17', '8B', 'hep',
                  'atmNu_SURF_avg']
ER_SOURCES    = ['pp', 'Be7_384', 'Be7_861', 'pep', 'N13', 'O15', 'F17']
BKGD_SOURCES  = ['Kr85', 'Rn222', 'nubb']

# ---------------------------------------------------------------------------
# NR Xe ideal
# ---------------------------------------------------------------------------
target = 'Xe'
fig, ax = plt.subplots(figsize=(6, 4))
for source in NR_SOURCES_XE:
    Er, rate, _ = nr_rate(source, target)
    color, ls   = get_colors(source)
    ax.loglog(Er, rate, color=color, ls=ls, label=get_official_pcle(source))
Er_sum, rate_sum = get_detected_spectrum(target=target, channel='NR',
                                         nu_sources=NR_SOURCES_XE, mode='ideal')
ax.loglog(Er_sum, rate_sum, color='black', ls='-', label='total')
ax = setup_cdfpdf_ax(ax, '', 'Nucleus recoil kinetic energy Er [keV]', '', True, 'pdf', 15, 15,
                     vlines=[], hlines=[], xlims=[0.001, 1000], ylims=[1e-5, 1e7], log=[0, 0])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=10)
save_name = os.path.join('figures', 'NR_'+target+'_ideal.pdf')
fig.savefig(save_name, bbox_inches='tight')

# ---------------------------------------------------------------------------
# ER Xe ideal
# ---------------------------------------------------------------------------
target = 'Xe'
fig, ax = plt.subplots(figsize=(6, 4))
for source in ER_SOURCES:
    Er, rate, _ = er_rate(source, target)
    color, ls   = get_colors(source)
    ax.loglog(Er, rate, color=color, ls=ls, label=get_official_pcle(source))
for source in BKGD_SOURCES:
    Er, rate = _read_background(source, target)
    if Er is not None:
        color, ls = get_colors(source)
        ax.loglog(Er, rate, color=color, ls=ls, label=get_official_pcle(source))
Er_sum, rate_sum = get_detected_spectrum(target=target, channel='ER',
                                         nu_sources=ER_SOURCES,
                                         bkgd_sources=[], mode='ideal')
ax.loglog(Er_sum, rate_sum, color='black', ls='-', label='total')
ax = setup_cdfpdf_ax(ax, '', 'Electron recoil kinetic energy Er [keV]', '', True, 'pdf', 15, 15,
                     vlines=[], hlines=[], xlims=[1e-3, 3000], ylims=[1e-5, 200], log=[0, 0])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=10)
save_name = os.path.join('figures', 'ER_'+target+'_ideal.pdf')
fig.savefig(save_name, bbox_inches='tight')
'''
# ---------------------------------------------------------------------------
# NR Ar ideal
# ---------------------------------------------------------------------------
target = 'Ar'
fig, ax = plt.subplots(figsize=(6, 4))
for source in NR_SOURCES_AR:
    Er, rate, _ = nr_rate(source, target)
    color, ls   = get_colors(source)
    ax.loglog(Er, rate, color=color, ls=ls, label=get_official_pcle(source))
Er_sum, rate_sum = get_detected_spectrum(target=target, channel='NR',
                                         nu_sources=NR_SOURCES_AR, mode='ideal')
ax.loglog(Er_sum, rate_sum, color='black', ls='-', label='total')
ax = setup_cdfpdf_ax(ax, '', 'Nucleus recoil kinetic energy Er [keV]', '', True, 'pdf', 15, 15,
                     vlines=[], hlines=[], xlims=[0.001, 1000], ylims=[1e-5, 1e8], log=[0, 0])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=10)
save_name = os.path.join('figures', 'NR_'+target+'_ideal.pdf')
fig.savefig(save_name, bbox_inches='tight')

# ---------------------------------------------------------------------------
# ER Ar ideal
# ---------------------------------------------------------------------------
target = 'Ar'
fig, ax = plt.subplots(figsize=(6, 4))
for source in ER_SOURCES:
    Er, rate, _ = er_rate(source, target)
    color, ls   = get_colors(source)
    ax.loglog(Er, rate, color=color, ls=ls, label=get_official_pcle(source))
for source in BKGD_SOURCES:
    Er, rate = _read_background(source, target)
    if Er is not None:
        color, ls = get_colors(source)
        ax.loglog(Er, rate, color=color, ls=ls, label=get_official_pcle(source))
Er_sum, rate_sum = get_detected_spectrum(target=target, channel='ER',
                                         nu_sources=ER_SOURCES,
                                         bkgd_sources=BKGD_SOURCES, mode='ideal')
ax.loglog(Er_sum, rate_sum, color='black', ls='-', label='total')
ax = setup_cdfpdf_ax(ax, '', 'Electron recoil kinetic energy Er [keV]', '', True, 'pdf', 15, 15,
                     vlines=[], hlines=[], xlims=[1e-3, 3000], ylims=[1e-5, 200], log=[0, 0])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=10)
save_name = os.path.join('figures', 'ER_'+target+'_ideal.pdf')
fig.savefig(save_name, bbox_inches='tight')
'''
print('Done.')
