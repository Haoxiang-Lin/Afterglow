# -*- coding: utf-8 -*-
import numpy as np

from scipy.optimize import fsolve, minimize_scalar
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d
from scipy.special import hyp2f1, gamma
def interp1d_loglog(x, y):
    return lambda xi: 10**(interp1d(np.log10(x), np.log10(y), kind='linear', fill_value='extrapolate')(np.log10(xi)))
def quad_loglog(func, a, b, *args, **kwargs):
    return quad(lambda logx: np.exp(logx) * func(np.exp(logx)), np.log(a), np.log(b), *args, **kwargs)

import astropy.units as u
from astropy.constants import c, m_e, m_p, e, h, sigma_T, M_sun
from astropy.cosmology import Planck15, z_at_value

from functools import lru_cache

from naima.models import (
    ExponentialCutoffBrokenPowerLaw,
    Synchrotron,
    InverseCompton,
    EblAbsorptionModel,
)

from naima.utils import trapz_loglog

import warnings

e = e.gauss
pcm = u.cm**(-3)
mec = m_e*c
mec2 = m_e*c**2

def unit_wrapper(*x):
    return [xx.value for xx in ([x] if isinstance(x, u.Quantity) else x)] * (x.unit if isinstance(x, u.Quantity) else x[0].unit)

@lru_cache(maxsize=None, typed=True)
def mu_obs(theta, phi, theta_obs): 
    return np.sin(theta)*np.cos(phi)*np.sin(theta_obs) + np.cos(theta)*np.cos(theta_obs)

@lru_cache(maxsize=None, typed=True)
def shock_dynamics(g0, s_ej, s_amb):
    u0, b0 = np.sqrt(g0**2-1), np.sqrt(1-1/g0**2)
    
    if np.isposinf(s_ej):
        return np.vectorize(lambda xi: u0 if xi < 1e-2 else (
            u0*xi**((s_amb-3)/2) if xi > 1e2 else 
            np.sqrt( ( (g0+1)/2*xi**(s_amb-3)*(np.sqrt(1+4*g0/(g0+1)*xi**(3-s_amb) + (2/(g0+1)*xi**(3-s_amb))**2 )-1) ) ** 2 - 1)
        ))

    ui = 100*u0
    eiso_u = lambda u: (max(u0, min(ui, u))**(-s_ej) - ui**(-s_ej)) / (u0**(-s_ej) - ui**(-s_ej))
    I = lambda u: s_ej/(s_ej+2) * u**(-s_ej-2) * (1+hyp2f1(-1/2, -1-s_ej/2, -s_ej/2, -u**2))
    miso_u = lambda u: (I(max(u0, min(ui, u))) - I(ui))/(u0**(-s_ej) - ui**(-s_ej))
    return np.vectorize(lambda xi: fsolve(lambda _u: xi**(3-s_amb)*(_u[0]/u0)**2 - eiso_u(_u[0]) + miso_u(_u[0])*(np.sqrt(_u[0]**2+1)-1), u0)[0])

@lru_cache(maxsize=None, typed=True)
def eats(t_obs, theta_obs, z, E0, g0, n0):
    s_ej = np.inf
    s_amb = 0
        
    u0, b0 = np.sqrt(g0**2-1), np.sqrt(1-1/g0**2)
    
    u_xi = shock_dynamics(g0, s_ej, s_amb)
    
    R_dec = (  ( (3-s_amb) * E0 / (4.*np.pi* n0 * m_p * c**2 * u0**2) ) ** (1/(3-s_amb))  ).to(u.pc)
    
    eats_calc = lambda xi: -np.cos(theta_obs)*xi + quad(lambda _xi: np.sqrt(u_xi(_xi)**2+1)/u_xi(_xi), 0, xi)[0]
    xiz = (c*t_obs/(1+z)/R_dec).cgs
    xi0 = b0*xiz/(1-b0*np.cos(theta_obs))
    xi = fsolve(lambda _xi: eats_calc(_xi[0]) - xiz, xi0)[0]
    
    Rb = xi*R_dec
    tb = (R_dec/c).cgs * quad(lambda _xi: np.sqrt(u_xi(_xi)**2+1)/u_xi(_xi), 0, xi)[0]
    Gsh = np.sqrt(u_xi(xi)**2+1)
    
    return Gsh, Rb, tb

@lru_cache(maxsize=None, typed=True)
def microphysics(Gsh, Rb, tb, n_amb, epsb, epse, pe, fe, electron_y):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    specific_heat_ratio = 4/3+1/Gsh
    compression_ratio = (specific_heat_ratio*Gsh+1)/(specific_heat_ratio-1)

    B = (np.sqrt(8*np.pi*epsb*compression_ratio*n_amb*m_p*c**2*(Gsh-1))).to((u.erg*pcm)**(1/2))
    Ne = fe * n_amb * 4/3*np.pi*Rb**3

    gm = (epse/fe*(pe-2)/(pe-1)*m_p/m_e*(Gsh-1)).cgs

    _gc = (6*np.pi*mec*Gsh/(sigma_T*B**2*tb)).cgs
    Y = (-1+np.sqrt(1+4*min(1, (gm/_gc)**(pe-2))*epse/epsb))/2
    Y *= float(electron_y)
    gc = _gc/(1+Y)
    
    g_syn_max = np.sqrt(6*np.pi*e/(sigma_T*B*(1+Y))).cgs
    g_esc_max = (e*B*Rb)/(12*Gsh*mec2).cgs
    
    g0 = max(2, min(gm, gc))
    g_break = max(2, gm, gc)
    p1 = pe if gm < gc else 2
    p2 = pe+1
    g_cutoff = min(g_syn_max, g_esc_max)
    beta_cutoff = 2 if g_syn_max < g_esc_max else 1
    
    # amp = (Ne/mec2/( g0/(p1-1) + g_break*(g_break/g0)**(-p1)*(p2-p1)/(p1-1)/(p2-1) )).to(1/u.eV)
    amp = (Ne/mec2/( g0/(p1-1) )).to(1/u.eV)
    
    electrons = ExponentialCutoffBrokenPowerLaw(
        amplitude = amp,
        e_0 = (g0*mec2).to(u.eV),
        e_break = (g_break*mec2).to(u.eV),
        alpha_1 = p1,
        alpha_2 = p2,
        e_cutoff = (g_cutoff*mec2).to(u.eV),
        beta = beta_cutoff,
    )
    
    return electrons, B

@lru_cache(maxsize=None, typed=True)
def emission(electrons, B, Rb, ssc):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    SYN = Synchrotron(
            electrons, 
            B=B.to((u.erg*pcm)**(1/2)).value * u.G, 
            Eemax=max(electrons.e_break, electrons.e_cutoff), 
            Eemin=electrons.e_0,
            nEed=50,
    )
    

    E_ph = np.logspace(-7, 9, 100) * u.eV
    L_syn = SYN.flux(E_ph, distance=0 * u.cm)
    phn_syn = L_syn / (4 * np.pi * Rb** 2 * c) * 2.24 * float(ssc)
    IC = InverseCompton(
        electrons,
        seed_photon_fields=[
            ["SSC", E_ph, phn_syn],
        ],
        Eemax=max(electrons.e_break, electrons.e_cutoff), 
        Eemin=electrons.e_0,
        nEed=50,
    )
    
    return SYN, IC

def tran_ssa(nu, electrons, B, Rb):
    p1 = electrons.alpha_1
    g0 = (electrons.e_0/mec2).cgs.value
    
    # eq.34 of Gould 1979, A&A, 76, 306
    Ke = electrons.amplitude*mec2
    nu_L = e*B/mec/(2*np.pi)
    fp = np.pi**(1/2) * 3**((p1+1)/2)/8 * gamma((3*p1+22)/12) * gamma((3*p1+2)/12) * gamma((p1+6)/4) / gamma((p1+8)/4)
    kappa = Ke * (2*np.pi*e/B) * (nu/nu_L)**(-(p1+4)/2) * fp
    tau_ssa = (2 * kappa / (4*np.pi*Rb**2)).cgs
    
    return np.where(tau_ssa < 1e-3, 1, 3*(1/2 + np.exp(-tau_ssa)/tau_ssa - (1-np.exp(-tau_ssa))/tau_ssa**2)/tau_ssa)

def tran_gg(e_an, SYN, IC, Rb):
    E_an_arr = np.logspace(int(np.log10(e_an.to(u.eV).value)), 14, 50)*u.eV
    L_an_arr = SYN.flux(E_an_arr, distance=0 * u.cm) + IC.flux(E_an_arr, distance=0*u.cm)
    phn_an_arr = L_an_arr / (4 * np.pi * Rb**2 * c) * 2.24
    n_gg = trapz_loglog(phn_an_arr.to(1/u.cm**3/u.eV).value, E_an_arr.to(u.eV).value) * u.cm**-3
    # n_gg = (e_an[0] * phn_an).to(1/u.cm**3)

    tau_gg = (2 * 0.2*sigma_T*n_gg * Rb).cgs

    return np.where(tau_gg < 1e-3, 1, 3*(1/2 + np.exp(-tau_gg)/tau_gg - (1-np.exp(-tau_gg))/tau_gg**2)/tau_gg)

@lru_cache(maxsize=None, typed=True)
def isoeq(e_obs, t_obs, theta_obs, z, E0, g0, n0, epsb, epse, pe, fe, ssa, gg, ebl, ssc, electron_y, beam_corr):
    s_ej = np.inf
    s_amb = 0
        
    if g0 <= 1 or E0 == 0:
        return 0 * u.erg/u.cm**2/u.s
    
    Gsh, Rb, tb = eats(t_obs, theta_obs, z, E0, g0, n0)
    n_amb = n0*Rb**-s_amb
    
    if Gsh <= 1.0:
        return 0 * u.erg/u.cm**2/u.s

    electrons, B = microphysics(Gsh, Rb, tb, n_amb, epsb, epse, pe, fe, electron_y)
    
    Doppler = 1/(Gsh*(1-np.sqrt(1-1/Gsh**2)*np.cos(theta_obs)))
    distance = Planck15.luminosity_distance(z)
    e_obs = unit_wrapper(e_obs)
    
    SYN, IC = emission(electrons, B, Rb, ssc)
    flx = Doppler**4 * SYN.sed((1+z)*e_obs/Doppler, distance)[0]
    flx += Doppler**4 * IC.sed((1+z)*e_obs/Doppler, distance)[0]
    
    if ssa:
        flx *= tran_ssa((1+z)*e_obs[0]/Doppler/h, electrons, B, Rb)

    if gg:
        flx *= tran_gg(Doppler*(mec2**2/((1+z)*e_obs/Doppler)), SYN, IC, Rb)
    
    if ebl:
        flx *= EblAbsorptionModel(redshift=z).transmission(e=e_obs)[0]
    
    if beam_corr:
        flx *= (1-np.cos(1/Gsh))/2
    
    return flx


def gaussian(e_obs, t_obs, theta_obs, z, E0=1e52*u.erg, g0=1000, theta_j=0.1, n0=0.01*u.cm**-3, epsb=0.0001, epse=0.1, pe=2.5, fe=0.1, ssa=True, gg=True, ebl=True, ssc=True, electron_y=True):
    
    beam_corr = False
    E_theta = lambda theta: E0 * np.exp(-theta**2/theta_j**2/2)
    g_theta = lambda theta: 1 + (g0-1) * np.exp(-theta**2/theta_j**2/2)
    
    integrand = lambda theta, phi: np.sin(theta)/(4*np.pi) * isoeq(e_obs, t_obs, np.arccos(mu_obs(theta, phi, theta_obs)), z, E_theta(theta), g_theta(theta), n0, epsb, epse, pe, fe, ssa, gg, ebl, ssc, electron_y, beam_corr).to(u.erg/u.cm**2/u.s).value
    return dblquad(integrand, -np.pi, np.pi, 0, np.pi/2)[0] * u.erg/u.cm**2/u.s

def gaussian_domain(e_obs, t_obs, theta_obs, z, E0=1e52*u.erg, g0=1000, theta_j=0.1, n0=0.01*u.cm**-3, epsb=0.0001, epse=0.1, pe=2.5, fe=0.1, ssa=True, gg=True, ebl=True, ssc=True, electron_y=True):
    
    beam_corr = False
    E_theta = lambda theta: E0 * np.exp(-theta**2/theta_j**2/2)
    g_theta = lambda theta: 1 + (g0-1) * np.exp(-theta**2/theta_j**2/2)

    def logflx(theta, phi):
        flx = isoeq(e_obs, t_obs, np.arccos(mu_obs(theta, phi, theta_obs)), z, E_theta(theta), g_theta(theta), n0, epsb, epse, pe, fe, ssa, gg, ebl, ssc, electron_y, beam_corr).to(u.erg/u.cm**2/u.s).value
        if flx == 0:
             return -np.inf
        else:
            return np.log(flx)

    theta_range = np.linspace(-np.pi/2, np.pi/2, 50)
    theta_range = theta_range[~np.isinf(np.array([-logflx(theta, 0) for theta in theta_range]))]
    if len(theta_range) == 0:
        int_l, int_r, int_h = 0, np.pi/2, np.pi
        print('Warning: cannot capture domain due to extremely low afterglow flux.')
        return [int_l, int_r, int_h]
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

    phi_range = np.linspace(-np.pi, np.pi, 50)
    phi_range = phi_range[~np.isinf(np.array([logflx(int_c, phi) for phi in phi_range]))]
    int_h = minimize_scalar(lambda phi: abs(shadow**2-(logflx(int_c, phi)-logflx_max)**2), method='Bounded', bounds=(phi_range[0], phi_range[-1]))
    if int_h.fun > abs(shadow**2-1):
        int_h = np.pi
    else:
        int_h = abs(int_h.x)

    return [int_l, int_r, int_h]

def gaussian_smart(e_obs, t_obs, theta_obs, z, E0=1e52*u.erg, g0=1000, theta_j=0.1, n0=0.01*u.cm**-3, epsb=0.0001, epse=0.1, pe=2.5, fe=0.1, ssa=True, gg=True, ebl=True, ssc=True, electron_y=True, domain=None, int_step=4):
    if domain is None:
        int_l, int_r, int_h = gaussian_domain(e_obs, t_obs, theta_obs, z, E0, g0, theta_j, n0, epsb, epse, pe, fe)
    else:
        int_l, int_r, int_h = domain
    # int_l, int_r, int_h = 0, min(theta_j, 1/eats(t_obs, theta_obs, z, E0, g0, n0)[0]), np.pi
    
    beam_corr = False
    E_theta = lambda theta: E0 * np.exp(-theta**2/theta_j**2/2)
    g_theta = lambda theta: 1 + (g0-1) * np.exp(-theta**2/theta_j**2/2)
    
    integrand = lambda theta, phi: np.sin(theta)/(4*np.pi) * isoeq(e_obs, t_obs, np.arccos(mu_obs(theta, phi, theta_obs)), z, E_theta(theta), g_theta(theta), n0, epsb, epse, pe, fe, ssa, gg, ebl, ssc, electron_y, beam_corr).to(u.erg/u.cm**2/u.s).value
    theta_l = np.linspace(int_l, int_r, int_step)
    phi_l = np.linspace(-int_h, int_h, int_step*2)
    return np.trapz([np.trapz([integrand(theta, phi) for phi in phi_l], phi_l) for theta in theta_l], theta_l) * u.erg/u.cm**2/u.s