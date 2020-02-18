# -*- coding: utf-8 -*-
import numpy as np

from scipy.optimize import fsolve, minimize_scalar
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
def interp1d_loglog(x, y):
    return lambda xi: 10**(interp1d(np.log10(x), np.log10(y), kind='linear', fill_value='extrapolate')(np.log10(xi)))

import astropy.units as u
from astropy.constants import c, m_e, m_p, e, h, sigma_T, M_sun
from astropy.cosmology import Planck15, z_at_value

from naima.models import (
    ExponentialCutoffBrokenPowerLaw,
    Synchrotron,
    InverseCompton,
    EblAbsorptionModel,
)

import warnings

__all__ = [
    "tophat",
    "structure",
    "int_range",
    "structure_smart",
]

e = e.gauss
pcm = u.cm**(-3)
mec = m_e*c
mec2 = m_e*c**2

# Other parameters
# alpha_stf: power index of radial velocity stratification, eiso_u ~ u^(-alpha_stf)
# n_amb: ambient particle density
# eps_B, eps_e: microphysical parameters
# p_e: power index of electron distribution
# zeta_e: accelerated number fraction of (external) electrons
alpha_stf, n_amb, eps_B, eps_e, p_e, zeta_e = np.inf, 0.01*pcm, 0.01, 0.1, 2.2, 0.1

mu = lambda theta, phi, theta_obs: np.sin(theta)*np.cos(phi)*np.sin(theta_obs) + np.cos(theta)*np.cos(theta_obs)

# return isotropic equivalent energy flux [erg s^-1 cm^-2] from the coordinate (theta, phi) 
# on the equal time arrival surface of an infinitestimal tophat jet.
# E0, g0: initial isotropic energy & Lorentz factor
# energy: observed energy (e.g. 3*u.GHz*h); time: observed time after merger
def tophat(E0, g0, energy, time, theta_obs, D_L):
    mu_obs = np.cos(theta_obs)
    z = z_at_value(Planck15.luminosity_distance, D_L)
    if isinstance(energy.to_value(u.eV), float):
        energy = np.array([energy.to_value(u.eV)])*u.eV

    # calc: R_b, t_b, Gamma_sh
    u0, b0 = np.sqrt(g0**2-1), np.sqrt(1-1/g0**2)
    if g0 == 1.0:
        return 0 * u.erg/u.cm**2/u.s

    R_dec = ((3.*E0/(4.*np.pi*n_amb*m_p*c**2*(g0**2-1)))**(1/3)).to(u.pc)
    xi_z = (c*time/(1+z)/R_dec).cgs

    if np.isposinf(alpha_stf):
        u_xi = lambda xi: u0 if xi < 1e-1 else (
            u0*xi**(-3/2) if xi > 1e1 else 
            np.sqrt(((g0+1)/2*xi**(-3)*(np.sqrt(1+4*g0/(g0+1)*xi**3+4/(g0+1)**2*xi**6)-1))**2-1)
        )
    else:
        ui = np.inf
        eiso_u = lambda u: (max(u0, min(ui, u))**(-alpha_stf) - ui**(-alpha_stf)) / (u0**(-alpha_stf) - ui**(-alpha_stf))
        I = lambda u: alpha_stf/(alpha_stf+2) * u**(-alpha_stf-2) * (1+hyp2f1(-1/2, -1-alpha_stf/2, -alpha_stf/2, -u**2))
        miso_u = lambda u: (I(max(u0, min(ui, u))) - I(ui))/(u0**(-alpha_stf) - ui**(-alpha_stf))
        u_xi = lambda xi: fsolve(lambda _u: xi**3*(_u[0]/u0)**2 - eiso_u(_u[0]) + miso_u(_u[0])*(np.sqrt(_u[0]**2+1)-1), u0)[0]

    eats = lambda xi: -mu_obs*xi + quad(lambda _xi: np.sqrt(u_xi(_xi)**2+1)/u_xi(_xi), 0, xi)[0]
    xi = fsolve(lambda _xi: eats(_xi[0]) - xi_z, b0*xi_z/(1-b0*mu_obs))[0]

    R_b = xi*R_dec
    t_b = (R_dec/c).cgs * quad(lambda _xi: np.sqrt(u_xi(_xi)**2+1)/u_xi(_xi), 0, xi)[0]
    Gamma_sh = np.sqrt(u_xi(xi)**2+1)

    if Gamma_sh == 1.0:
        return 0 * u.erg/u.cm**2/u.s


    # calc: electron distribution
    specific_heat_ratio = 4/3+1/Gamma_sh
    compression_ratio = (specific_heat_ratio*Gamma_sh+1)/(specific_heat_ratio-1)

    B = (np.sqrt(8*np.pi*eps_B*compression_ratio*n_amb*m_p*c**2*(Gamma_sh-1))).to((u.erg*pcm)**(1/2))
    Ne = zeta_e * 4/3*np.pi*R_b**3 * n_amb

    gm = (eps_e/zeta_e*(p_e-2)/(p_e-1)*m_p/m_e*(Gamma_sh-1)).cgs

    _gc = 6*np.pi*mec*Gamma_sh/(sigma_T*B**2*t_b)
    Y = (-1+np.sqrt(1+4* min(1, (gm/_gc)**(p_e-2)) *eps_e/eps_B))/2
    gc = _gc.cgs/(1+Y) # rough estimate from Sari & Esin (2001)

    e_syn_max = (min(np.sqrt(3*e/(sigma_T*B*(1+Y))), (e*B*R_b)/(12*2*np.pi*Gamma_sh*mec2))*mec2).to(u.TeV)

    electrons = ExponentialCutoffBrokenPowerLaw(
        amplitude = (Ne*(p_e-1)/(max(2., gm)*mec2)*min(1., (gm/2.)**(p_e-1)) if gm<gc else Ne/(max(2., gc)*mec2)*min(1., gc/2.)).to(1/u.eV),
        e_0 = (max(2., min(gm, gc))*mec2).to(u.TeV),
        e_break = (max(2., max(gm, gc))*mec2).to(u.TeV),
        alpha_1 = p_e if gm<gc else 2.,
        alpha_2 = p_e+1,
        e_cutoff = max(((max(2., min(gm, gc))+1)*mec2).to(u.TeV), e_syn_max),
        beta=2.0,
    )
    
    
    # calc: syn+ic flux
    SYN = Synchrotron(
            electrons, 
            B=B.to_value((u.erg*pcm)**(1/2)) * u.G, 
            Eemax=electrons.e_cutoff, 
            Eemin=electrons.e_0,
            nEed=50,
    )

    '''E_ph = np.logspace(-7, 14, 22) * u.eV
    L_syn = self.SYN.flux(E_ph, distance=0 * u.cm)
    phn_syn = L_syn / (4 * np.pi * self.R ** 2 * c) * 2.24

    IC = InverseCompton(
        self.electrons,
        seed_photon_fields=[
            #"CMB",
            #["FIR", 70 * u.K, 0.5 * u.eV / u.cm ** 3],
            #["NIR", 5000 * u.K, 1 * u.eV / u.cm ** 3],
            ["SSC", E_ph, phn_syn],
        ],
        Eemax=self.electrons.e_cutoff, 
        Eemin=self.electrons.e_0,
        nEed=50,
    )'''

    Doppler = 1/(Gamma_sh*(1-np.sqrt(1-1/Gamma_sh**2)*mu_obs))
    flx = Doppler**4 * SYN.sed((1+z)*energy/Doppler, D_L)[0]
    tran = EblAbsorptionModel(redshift=z).transmission(e=energy)[0]
    return (flx * tran).to(u.erg/u.cm**2/u.s)

# Gaussian jet
# Eiso0 = lambda theta: 10**(51) * u.erg * np.exp(-(theta/0.1)**2/2)
# Gamma0 = lambda theta: 1 + (300-1) * np.exp(-(theta/0.1)**2/2)

# integrate total energy flux [erg s^-1 cm^-2] of a structured jet
def structure(Eiso0, Gamma0, energy, time, theta_obs, D_L):
    
    integrand = lambda theta, phi: np.sin(theta)/(4*np.pi) * tophat(Eiso0(theta), Gamma0(theta), energy, time, np.arccos(mu(theta, phi, theta_obs)), D_L).to_value(u.erg/u.cm**2/u.s)
    return dblquad(integrand, -np.pi, np.pi, 0, np.pi)[0] * u.erg/u.cm**2/u.s

# find effective integral range (WARNING: still very empirical)
def int_range(Eiso0, Gamma0, energy, time, theta_obs, D_L):
    
    warnings.simplefilter('ignore', RuntimeWarning)

    def logflx(theta, phi):
        return np.log(tophat(Eiso0(theta), Gamma0(theta), energy, time, np.arccos(mu(theta, phi, theta_obs)), D_L).to_value(u.erg/u.cm**2/u.s))

    theta_test = np.linspace(-np.pi/2, np.pi/2, 50)
    logflx_test = np.array([-logflx(theta, 0) for theta in theta_test])
    theta_range = theta_test[~np.isinf(logflx_test)]
    int_c = minimize_scalar(lambda theta: -logflx(theta, 0), method='Bounded', bounds=(theta_range[0], theta_range[-1]))
    int_c = abs(int_c.x)

    logflx_max = logflx(int_c, 0)
    shadow = 2

    int_l = minimize_scalar(lambda theta: abs(shadow**2-(logflx(theta, 0)-logflx_max)**2), method='Bounded', bounds=(0., int_c))
    int_l = int_l.x

    int_r_lim = np.pi/2
    int_r_iter = 0
    int_r = minimize_scalar(lambda theta: abs(shadow**2-(logflx(theta, 0)-logflx_max)**2), method='Bounded', bounds=(int_c, min(theta_range[-1], int_r_lim)))
    while int_r.fun > shadow**2:
        int_r_iter += 1
        int_r_lim = (int_r_lim+int_c)/2
        int_r = minimize_scalar(lambda theta: abs(shadow**2-(logflx(theta, 0)-logflx_max)**2), method='Bounded', bounds=(int_c, min(theta_range[-1], int_r_lim)))
        if int_r_iter > 50:
            break
    int_r = int_r.x

    phi_test = np.linspace(-np.pi, np.pi, 50)
    logflx_test = np.array([logflx(int_c, phi) for phi in phi_test])
    phi_range = phi_test[~np.isinf(logflx_test)]
    int_h = minimize_scalar(lambda phi: abs(shadow**2-(logflx(int_c, phi)-logflx_max)**2), method='Bounded', bounds=(phi_range[0], phi_range[-1]))
    if int_h.fun > abs(shadow**2-1):
        int_h = np.pi
    else:
        int_h = abs(int_h.x)
        
    warnings.resetwarnings()
    
    return [int_l, int_r, int_h]

# integrate total energy flux [erg s^-1 cm^-2] of a structured jet (smart integral)
def structure_smart(Eiso0, Gamma0, energy, time, theta_obs, D_L, int_config):
    int_l, int_r, int_h = int_config
    
    integrand = lambda theta, phi: np.sin(theta)/(4*np.pi) * tophat(Eiso0(theta), Gamma0(theta), energy, time, np.arccos(mu(theta, phi, theta_obs)), D_L).to_value(u.erg/u.cm**2/u.s)
    theta_l = np.linspace(int_l, int_r, 4)
    phi_l = np.linspace(-int_h, int_h, 8)
    return np.trapz([np.trapz([integrand(theta, phi) for phi in phi_l], phi_l) for theta in theta_l], theta_l) * u.erg/u.cm**2/u.s
