import numpy as np
import os

import random
import math
import matplotlib.pyplot as plt
import string
import astropy.units as u
import astropy.constants as const
from scipy.special import spherical_jn, erf
from scipy.special import gamma

class delxsec_dEr_nucleus():
    def __init__(self, E_neu, Er, A, Z, 
                 unit_convert = (const.hbar*const.c).to(u.MeV*u.fm),
                 unit_dxecdEr = u.cm**2/u.keV, 
                 s = 0.9*u.fm, r_0 = 0.52 *u.fm, 
                 sin2thetaw =0.231,#https://arxiv.org/abs/1810.05606, 
                 Gf = 1.1663787e-5*u.GeV**(-2)
                 ):
        """
        E_neu: incoming neutrino energy
        Er: neutrino-nucleus recoil kinetic energy
        """

        
        self.Er = Er
        self.E_neu = E_neu
        
        self.unit_convert = unit_convert
        self.unit_dxecdEr = unit_dxecdEr
        
        self.A = A
        self.Z = Z
        self.N = self.A - self.Z
        self.sin2thetaw = sin2thetaw
        self.Gf = Gf
        
        self.m_N = self.A * (const.m_n * const.c**2).to(u.GeV)
        
        #helm form factor https://www.tir.tw/phys/hep/dm/amidas/equations/eq-FQ_Helm.html
        self.s = s #nuclear skin thickness
        self.r_0 = r_0
        self.R_A = (1.23* self.A**(1/3) - 0.6)*u.fm 
        self.R = np.sqrt(self.R_A**2 +(7/3) * np.pi**2 * self.r_0**2 - 5 * self.s**2) #effective nuclear radius
        
        
        self.q = np.sqrt(2*self.m_N*self.Er)#momentum transfer
        #print(self.q, self.R)
        
    def Q_w(self):#weak nuclear charge
        return  self.N - (1 - 4*self.sin2thetaw ) * self.Z
    
    def F2_Q(self):
        # first Born (plane wave) approximation
        R_nucleus = self.R
        x = self.q * R_nucleus
        
        x_dim =  (x/ self.unit_convert ).decompose()
        if x_dim.unit !=(u.m/u.m):
            print('dimension error')
            return 0
        exponential_term = x*self.s/R_nucleus
        exponential_term_dim = (exponential_term / self.unit_convert).decompose()
        F2_q = (3 * spherical_jn(1, x_dim.value) / (x_dim.value))**2 * np.exp(-(exponential_term_dim.value)**2) 

        return F2_q

    def dxsec_dEr(self):
        #check the kinematics
        if self.E_neu <= np.sqrt(self.Er * self.m_N / 2) : return 0*self.unit_dxecdEr

        F2_q = self.F2_Q()
        Qw = self.Q_w()
        dxsec_dEr = self.Gf**2 / 4/np.pi * Qw**2 * self.m_N * (1 - (self.m_N * self.Er/2/self.E_neu**2).decompose()
                                                         ) * F2_q
        
        dxsec_dEr_cm2perkeV = (dxsec_dEr * self.unit_convert**2).to(self.unit_dxecdEr)
        return dxsec_dEr_cm2perkeV


class delxsec_dEr_lepton():
    def __init__(self, E_neu, Er, nu_flavor, 
                 unit_convert = (const.hbar*const.c).to(u.MeV*u.fm),
                 unit_dxecdEr = u.cm**2/u.keV, 
                 sin2thetaw =0.231,#https://arxiv.org/abs/1810.05606, 
                 Gf = 1.1663787e-5*u.GeV**(-2)
                 ):
        """
        E_neu: incoming neutrino energy
        Er: neutrino-nucleus recoil kinetic energy
        """

        
        self.Er = Er
        self.E_neu = E_neu
        
        self.unit_convert = unit_convert
        self.unit_dxecdEr = unit_dxecdEr
        
        self.sin2thetaw = sin2thetaw
        self.Gf = Gf
        self.nu_flavor = nu_flavor

    #textbook physics of neutrinos and applications to astrophysics 福来正孝＆柳田勉
    #p92 3.10
    def get_ga(self):
        #axial coupling
        if self.nu_flavor == 'numutau':        
            ga = -1/2
        elif self.nu_flavor  == 'nue':
            ga=-1/2+1
        return  ga 
    
    def get_gv(self):
        # vectorial coupling
        if self.nu_flavor == 'numutau':        
            gv = 2 * self.sin2thetaw - 1/2
        elif self.nu_flavor  == 'nue':
            gv = 2 * self.sin2thetaw - 1/2 + 1
        return gv
    
        
        
    def dxsec_dEr(self):
        #check the kinematics
        me = (const.m_e*const.c**2).to(u.MeV)
        if self.E_neu <= 0.5 * (self.Er + np.sqrt( self.Er *(self.Er+2*me) 
                                                 ) 
                               ): return 0*self.unit_dxecdEr

        gv = self.get_gv()
        ga = self.get_ga()
        
        dxsec_dEr = self.Gf**2 *me/2/np.pi*((gv+ga)**2 + 
                                            (gv-ga)**2 * (1-self.Er/self.E_neu)**2 + 
                                            (ga**2-gv**2) * (me*self.Er/self.E_neu**2) 
                                           )
        
        dxsec_dEr_cm2perkeV = (dxsec_dEr * self.unit_convert**2).to(self.unit_dxecdEr)
        
        return dxsec_dEr_cm2perkeV


class nucapxsec_Bahcoll():
    def __init__(self, Z, A, E_neu, Q, logft, I_initial, I_final, 
                 sigma0_Const = 1.206e-42, # eqn11
                 unit_convert = (const.hbar*const.c).to(u.MeV*u.fm),
                 unit_xsec = u.cm**2,
                 mode=1,
                 alpha_em = 1/137.036):
        #https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.50.881 
        """
        E_neu: incoming neutrino energy
        Z: Cs eqn9 https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.50.881 
        """
        
        self.E_neu = E_neu
        self.Q = Q
        
        self.unit_convert = unit_convert
        self.unit_xsec = unit_xsec

        self.alpha_em = alpha_em
        
        #calculate logft https://www.nndc.bnl.gov/logft/
        self.logft = logft#ft1/2 I→I',  Cs electron capture, I = Cs, I' = Xe
        self.mode = mode
        
        self.I_initial = I_initial
        self.I_final = I_final
        
        self.Z = Z
        self.A = A
        self.m_e = (const.m_e*const.c**2).to(u.MeV)
        self.sigma0_Const = sigma0_Const

 
    def get_E_final(self):
        E_final = -self.Q + self.E_neu + self.m_e
        return 0*E_final.unit if E_final<0 else E_final
      
    def get_P_final(self):
        E_final = self.get_E_final()
        
        if E_final<self.m_e:
            
            P_final = 0*E_final.unit
        else:
            P_final = np.sqrt(E_final**2 - self.m_e**2)
        return P_final
    
    
    def get_R(self):
        
        #eqn20 https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.50.881
        R_JNB = (((2.9080 * self.A**(1/3) + 6.0910 * self.A**(-1/3) - 5.3610/self.A
                  )*1e-3 * const.hbar/(const.m_e*const.c)
                 ).to(u.fm)/self.unit_convert).to(1/u.MeV)

        return R_JNB
    
      
    def get_Fermi_function(self):
        # https://arxiv.org/pdf/nucl-th/0311022.pdf
        
        E_final = self.get_E_final()
        p_final = self.get_P_final() 
        Z_f = self.Z * self.mode

        R = self.get_R()
        gamma0 = np.sqrt(1 - (self.alpha_em*Z_f)**2) # eqn19 p885
        if p_final==0:
            return 0.
        else:
            eta = (self.alpha_em * Z_f * E_final / p_final).decompose()# eqn19 p885
            
            #eqn19 
            F_Z_we = 2*(1 + gamma0) * (2*p_final*R)**(-2*(1-gamma0)) * np.abs(gamma(complex(gamma0,eta)))**2/np.abs(gamma(2.*gamma0+1.))**2
            
            #eqn21
            F_Z_we_avg = F_Z_we / (1 - 2/3 * (1-gamma0))
            
            
            return F_Z_we_avg
    
    def get_sigma0(self):
        #eqn 11 p883 https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.50.881 
        
        ft = 10**self.logft
        Z_f = self.Z * self.mode
        I_prime = self.I_final
        I = self.I_initial
        sigma_0 = (self.sigma0_Const / ft * (2*I_prime+1)/(2*I+1) * self.Z * u.cm**2).to(self.unit_xsec)
        return sigma_0
    
    def get_G(self):
        
        #eqn 12 p883 https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.50.881 
        E_final = self.get_E_final() #final electron energy 
        p_final = self.get_P_final() #final electron momentum  
        F = self.get_Fermi_function()
        
        
        G = F * (p_final/self.m_e).decompose() / 2/np.pi/self.alpha_em/self.Z/(E_final/self.m_e).decompose()
        return G
    
    def xsec(self):
        #eqn10 p883 https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.50.881 
        
        G = self.get_G()
        E_final = self.get_E_final()
        sigma_0 = self.get_sigma0()
        xsec = sigma_0 * G * ((E_final/self.m_e).decompose())**2
        xsec_cm2 = xsec.to(self.unit_xsec)
                
        return xsec_cm2
    