import numpy as np
import os

import random
import math
import matplotlib.pyplot as plt
import string
import astropy.units as u
import astropy.constants as const

# The minimum and maximum recoil energy Er
def momentum(e, m):
    return np.sqrt(e**2 - m**2)

def ermax(echi, mchi, dE, mA):
    """
    DM-nucleus scattering
    echi: DM energy
    mchi: DM mass
    dE: excitation energy
    mA: nuclear mass
    return: maximum recoil energy
    """
    if echi <= mchi + dE: 
        print('echi <= mchi + dE') 
        return 0*dE.unit

    vchi = np.sqrt(echi**2 - mchi**2) / echi
    if vchi <= 0.01: # non-relativistic
        #eqn2 p2 https://arxiv.org/pdf/1211.7222
        #print('non-relativistic')
        mu_xN = mchi * mA / (mchi + mA) # reduced mass
        return 2 * mu_xN**2 * vchi**2 / mA

    pchi = momentum(echi, mchi)
    pchip = momentum(echi - dE, mchi)
    return (pchi + pchip)**2 / (2 * mA)

def ermin(echi, mchi, dE, mA):
    """
    DM-nucleus scattering
    echi: DM energy　
    mchi: DM mass
    dE: excitation energy
    mA: nuclear mass
    return: minimum recoil energy
    """
    if echi <= mchi + dE: 
        print('echi <= mchi + dE') 
        return 0*dE.unit

    vchi = np.sqrt(echi**2 - mchi**2) / echi
    if vchi <= 0.01: 
        #print('non-relativistic')
        return 0 # non-relativistic

    pchi = momentum(echi, mchi)
    pchip = momentum(echi - dE, mchi)
    return (pchi - pchip)**2 / (2 * mA) #momentum transfer 


def ermax_el(E_neu, M_target):
    Er_max = 2 * E_neu**2 / (M_target + 2 * E_neu)
    return Er_max

def ermin_el(E_neu, M_target):
    return 0*u.keV

def ermax_CEVENS(E_neu, M_N):
    Er_max = 2 * E_neu**2 / (M_N + 2 * E_neu)
    return Er_max

def ermin_CEVENS(E_neu, M_N):
    return 0*u.keV

def E_neu_min_CEVENS(E_r, M_N):
    return np.sqrt(M_N * E_r / 2)
