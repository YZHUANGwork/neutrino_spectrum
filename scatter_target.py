import numpy as np
import os
import astropy.units as u
import astropy.constants as const

def get_target_dNT_dM(A, abundance = 1.):
    #p14 https://arxiv.org/pdf/2104.12785
    M_u = 1*u.g/u.mol 
    
    dN_T_dM = const.N_A/M_u/A*abundance
    return dN_T_dM

def get_target_info(target):
    if target =='Xe':
        A = 131.293
        Z = 54
        
    elif target =='Ar':
        A = 39.948
        Z = 18
        
    elif target =='He':
        A = 4       
        Z = 2
    return A, Z


def get_binding_energy(target):
    if target == 'Xe':
        e_bind = np.array([34563]*2 + [5454.8]*2 + [4891.4]*6 + [1148]*2 + [959.78]*6 + [681.02]*10 + [214.63]*2 + [153.48]*6 + [
    68.146]*10 +[23.39]*2 + [12.563]*6)*(u.eV)
    
    elif target == 'Ar':
        e_bind = np.array([3206.2]*2 + [324.2]*2 + [247.74]*6 + [29.24]*2 + [15.76]*6)*(u.eV)
    else:
        print('not applicable')
        e_bind = 1.
    return e_bind


#neutrino capture process 
def get_logft(A):
    #https://www.nndc.bnl.gov/logft/
    #Cs EC/β+
    if A==131:
        #https://atom.kaeri.re.kr/cgi-bin/nuclide?nuc=Cs131
        logft = 5.54
        I = 5/2
        T_half = 9.69*u.day
        Q_val = 352*u.keV
    elif A==136:
        #https://atom.kaeri.re.kr/cgi-bin/nuclide?nuc=Cs136
        logft = 7.6
        I = 5
        T_half = 13.2*u.day
        Q_val = 2548*u.keV
    return logft, Q_val, I


def get_thrd_abundance(A, nuelci = 'Xenon'):
    #https://physics.nist.gov/PhysRefData/Handbook/Tables/xenontable1.htm
    if A == 124:
        E_nue_thrd = 5.93*u.MeV
        abundance = 0.095/100
    elif A == 126:
        E_nue_thrd = 4.80*u.MeV
        abundance = 0.089/100
    elif A == 128:
        E_nue_thrd = 3.93*u.MeV
        abundance = 1.9/100
    elif A == 129:
        E_nue_thrd =  1.20*u.MeV
        abundance =  26.4/100
    elif A == 130:
        E_nue_thrd =  2.98*u.MeV
        abundance =  4.1/100
    elif A == 131:
        E_nue_thrd =  0.355*u.MeV
        abundance =   21.2/100
    elif A == 132:
        E_nue_thrd =  2.12*u.MeV
        abundance =   26.9/100
    elif A == 134:
        E_nue_thrd =  1.23*u.MeV
        abundance =   10.4/100
    elif A == 136:
        E_nue_thrd =  0.0903*u.MeV
        abundance =   8.9/100
    return E_nue_thrd, abundance