import numpy as np
import os

import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

def setup_cdfpdf_ax(ax, title, xlabel, ylabel, gridTF, cdfpdf, label_size, font_size, 
                    vlines = [0], hlines = [0], xlims = [0,0], ylims = [0,0], log = [0,0]):
    ax.yaxis.set_tick_params(labelsize=label_size) 
    ax.xaxis.set_tick_params(labelsize=label_size)  
    ax.set_title(title, fontsize = font_size)
    ax.grid(gridTF)
    if cdfpdf in ['cdf']:
        ylabel = 'Event rate '+ r'[ton$^{-1}$ yr$^{-1}$]'
    elif cdfpdf in ['pdf']:
        ylabel = r'$\dfrac{d{\cal R}}{dE_{r}}$  '+ '[ton$^{-1}$ yr$^{-1}$ '+'keV'+'$^{-1}$]'
    elif cdfpdf in ['pmf']:
        ylabel = r'$d{\cal R}(E_{r})$  '+ '[ton$^{-1}$ yr$^{-1}$ '+']'
    ax.set_xlabel(xlabel, fontsize = label_size)
    ax.set_ylabel(ylabel, fontsize = label_size)
    
    if len(xlims)>0:
        if  xlims[0] != xlims[1]:
            ax.set_xlim(xlims[0], xlims[1])
    if len(ylims)>0:
        if ylims[0] != ylims[1]:
            ax.set_ylim(ylims[0], ylims[1])
    if len(vlines)>0:
        for vline in vlines:
            ax.axvline(x = vline, lw = 3, ls = '--', color = 'black')
    if len(hlines)>0:
        for hline in hlines:
            ax.axhline(y = hline, lw = 3, ls = '--', color = 'black')
    if log[0]!=0:
        ax.set_xscale('log')
    if log[1]!=0:
        ax.set_yscale('log')
    return ax 

def get_official_pcle(pcle):
    if pcle == 'Be7_861':
        return r'$^{7}$Be 861keV'
    elif pcle == 'Be7_384':
        return r'$^{7}$Be 384keV'
    elif pcle == 'Be7':
        return r'$^{7}$Be'
    elif pcle == 'N13':
        return r'$^{13}$N'
    elif pcle == 'O15':
        return r'$^{15}$O'
    elif pcle == 'F17':
        return r'$^{17}$F'
    elif pcle == 'nubb':
        return r'2$\nu\beta\beta$'
    elif pcle == 'atmNu_SURF_avg':
        return 'Atm SURF'
    elif pcle == 'Kr85':
        return r'$^{85}$Kr'
    elif pcle == 'Rn222':
        return r'$^{222}$Rn'
    elif pcle == '8B':
        return r'$^{8}$B'
    else:
        return pcle

def get_colors(pcle):
    pcles = ['pp' ,'N13','O15' ,'F17', 'pep', 'Be7_861', 'Be7_384','nubb', 'Rn222', 'Kr85',
            'hep', 'dsnb','8B', 'atmNu_SURF_avg']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'yellowgreen', 'turquoise', 'grey', 
              'coral','olive', 'teal', 'violet']
    line = '-'
    color_dict = dict(zip(pcles, colors))
    return color_dict.get(pcle), line

def get_isotope_color(isotope):
    if isotope == 'Xe129':
        color = 'yellowgreen'
    elif isotope == 'Xe131':
        color = 'forestgreen'
    elif isotope == 'Xe132':
        color = 'darkturquoise'
    elif isotope == 'Xe134':
        color = 'indigo'
    elif isotope == 'Xe136':
        color = 'magenta'
    elif isotope == 'Cs131':
        color = 'darkred'
    elif isotope == 'Cs136':
        color = 'goldenrod'

    return color
