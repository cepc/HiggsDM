#!/bin/env python3

__author__ = "Zhao-Huan Yu <yuzhaoh5@mail.sysu.edu.cn>"
__copyright__ = "Copyright (c) Zhao-Huan Yu"
__created__ = "[2021-04-27 Tue 16:01]" 

## Reference:
# [1] Arcadi, Djouadi, Raidal, 1903.03616, Phys. Rept.

import numpy as np                  # https://docs.scipy.org/doc/numpy/reference/
from scipy import optimize          # https://docs.scipy.org/doc/scipy/reference/
import matplotlib.pyplot as plt     # https://matplotlib.org/api/pyplot_api.html
import matplotlib.ticker as ticker  # https://matplotlib.org/api/ticker_api.html


# SM parameters
m_H = 125.10                        # GeV, H pole mass
Gamma_H_SM = 4.07e-3                # GeV, H decay width in the SM
G_F = 1.1663787e-5                  # GeV^-2, Fermi coupling constant
vev = (np.sqrt(2)*G_F)**-0.5        # GeV, Higgs vacuum expectation value v
m_p = 0.93827208816                 # Gev, proton mass
m_n = 0.93956542052                 # Gev, neutron mass

# Nucleon form factors, Eq. (48) in Ref. [1]
f_p_u = 2.08e-2                     # form factor f^p_u
f_p_d = 4.11e-2                     # form factor f^p_d
f_p_s = 4.3e-2                      # form factor f^p_s
f_p_Q = 2*(1 - f_p_u - f_p_d - f_p_s)/27  # form factor f^p_Q for Q = c, b, t
f_p = f_p_u + f_p_d + f_p_s + 3*f_p_Q     # sum of form factors for proton
f_n_u = 1.89e-2                     # form factor f^n_u
f_n_d = 4.51e-2                     # form factor f^n_d
f_n_s = f_p_s                       # form factor f^n_s
f_n_Q = 2*(1 - f_n_u - f_n_d - f_n_s)/27  # form factor f^p_Q for Q = c, b, t
f_n = f_n_u + f_n_d + f_p_s + 3*f_n_Q     # sum of form factors for neutron

# Units
GeVm1_to_cm = 1.973269804e-14       # GeV^-1 = 1.973269804e-14 cm


# Model for effective interaction of the Majorana DM particle chi and the SM Higgs field
# Lagrangian in Eq. (17) of Ref. [1]
class model_MajoranaDM:
    def __init__(self, m_chi, lam_Hchichi, Lambda=1e3):
        self.m_chi        =  m_chi        # GeV, DM particle mass
        self.lam_Hchichi  =  lam_Hchichi  # dimensionless h-chi-chi coupling
        self.Lambda       =  Lambda       # GeV, new physical scale, default = 1 TeV

    @property
    def width_inv(self):
        # Invisible Higgs decay width, H -> chi chi
        # 1/2 * Eq. (28) in Ref. [1]  (substitution: f -> chi, lambda_Hff -> lam_Hchichi)
        if self.m_chi < m_H/2:
            beta = np.sqrt(1 - 4*(self.m_chi/m_H)**2)
            wid = (self.lam_Hchichi*vev)**2*m_H*beta**3/(64*np.pi*self.Lambda**2)
        else:
            wid = 0
        return wid

    @property
    def BR_inv(self):
        # Invisible Higgs decay branching ratio
        # Eq. (52) in Ref. [1]
        return self.width_inv/(Gamma_H_SM + self.width_inv)

    @property
    def sigma_SI_chiN(self):
        # Spin-independent DM-nucleon scattering cross section in unit of cm^2
        # Eq. (51) in Ref. [1]
        # Adopt protons to represent nucleons
        return self.lam_Hchichi**2*m_p**4*self.m_chi**2*f_p**2/(4*np.pi*self.Lambda**2*m_H**4*(self.m_chi + m_p)**2)*GeVm1_to_cm**2


# Model for effective interaction of the scalar DM particle chi and the SM Higgs field
# Lagrangian in Eq. (17) of Ref. [1]
class model_scalarDM:
    def __init__(self, m_chi, lam_Hchichi):
        self.m_chi        =  m_chi        # GeV, DM particle mass
        self.lam_Hchichi  =  lam_Hchichi  # dimensionless h-chi-chi coupling

    @property
    def width_inv(self):
        # Invisible Higgs decay width, H -> chi chi
        # 1/2 * Eq. (28) in Ref. [1]  (substitution: S -> chi, lambda_HSS -> lam_Hchichi)
        if self.m_chi < m_H/2:
            beta = np.sqrt(1 - 4*(self.m_chi/m_H)**2)
            wid = (self.lam_Hchichi*vev)**2*beta/(128*np.pi*m_H)
        else:
            wid = 0
        return wid

    @property
    def BR_inv(self):
        # Invisible Higgs decay branching ratio
        # Eq. (52) in Ref. [1]
        return self.width_inv/(Gamma_H_SM + self.width_inv)

    @property
    def sigma_SI_chiN(self):
        # Spin-independent DM-nucleon scattering cross section in unit of cm^2
        # Eq. (51) in Ref. [1]
        # Adopt protons to represent nucleons
        return self.lam_Hchichi**2*m_p**4*f_p**2/(16*np.pi*m_H**4*(self.m_chi + m_p)**2)*GeVm1_to_cm**2


# Model for effective interaction of the vector DM particle chi and the SM Higgs field
# Lagrangian in Eq. (17) of Ref. [1]
class model_vectorDM:
    def __init__(self, m_chi, lam_Hchichi):
        self.m_chi        =  m_chi        # GeV, DM particle mass
        self.lam_Hchichi  =  lam_Hchichi  # dimensionless h-chi-chi coupling

    @property
    def width_inv(self):
        # Invisible Higgs decay width, H -> chi chi
        # 1/2 * Eq. (28) in Ref. [1]  (substitution: V -> chi, lambda_HVV -> lam_Hchichi)
        if self.m_chi < m_H/2:
            beta = np.sqrt(1 - 4*(self.m_chi/m_H)**2)
            wid = (self.lam_Hchichi*vev)**2*m_H**3*beta*(1 - 4*(self.m_chi/m_H)**2 + 12*(self.m_chi/m_H)**4)/(512*np.pi*self.m_chi**4)
        else:
            wid = 0
        return wid

    @property
    def BR_inv(self):
        # Invisible Higgs decay branching ratio
        # Eq. (52) in Ref. [1]
        return self.width_inv/(Gamma_H_SM + self.width_inv)

    @property
    def sigma_SI_chiN(self):
        # Spin-independent DM-nucleon scattering cross section in unit of cm^2
        # Eq. (51) in Ref. [1]
        # Adopt protons to represent nucleons
        return self.lam_Hchichi**2*m_p**4*f_p**2/(16*np.pi*m_H**4*(self.m_chi + m_p)**2)*GeVm1_to_cm**2


### Test
#print('f_p, f_n:', f_p, f_n)
#m = model_MajoranaDM(23., 0.01, 1e3)
#print('width_inv, BR_inv, sigma_SI_chiN:', m.width_inv, m.BR_inv, m.sigma_SI_chiN)


def lam_lim(model, BR_inv_lim, m_chi):
    # Calculate the limit of lam_Hchichi from the limit of BR_inv for fixed m_chi
    if m_chi >= m_H/2:
        return float('nan')
    else:
        def fun(lam):
            m = model(m_chi, lam)
            return m.BR_inv - BR_inv_lim
        lam_Hchichi = optimize.brentq(fun, 1e-9, 1e2)
        return lam_Hchichi

def inv_decay_limits(model, BR_inv_lim, m_chi_ar):
    # Calculate the limits of lam_Hchichi of sigmaSI_ar from the limit of BR_inv for an array m_chi_ar
    n = len(m_chi_ar)
    lam_ar = np.zeros(n)
    sigmaSI_ar = np.zeros(n)
    for i in range(n):
        lam_ar[i] = lam_lim(model, BR_inv_lim, m_chi_ar[i])
        if np.isnan(lam_ar[i]):
            sigmaSI_ar[i] = float('nan')
        else:
            m = model(m_chi_ar[i], lam_ar[i])
            sigmaSI_ar[i] = m.sigma_SI_chiN
            #print(m.BR_inv, lam_ar[i], m.sigma_SI_chiN)
    return lam_ar, sigmaSI_ar



###################################################################
# Plot setup
# print(plt.rcParams.keys()) will print out available rc parameters
def setup():
    plt.style.use('classic')
    fig, ax = plt.subplots()
    for stri in ['left', 'right', 'top', 'bottom']:
        ax.spines[stri].set_linewidth(1.2)
    plt.rc('font', size=20)
    plt.rc('axes', titlesize=20, titlepad=12)
    plt.xlabel('tmp', fontsize=22, labelpad=12)
    plt.ylabel('tmp', fontsize=22, labelpad=12)
    plt.tick_params(which='both', labelsize=16, length=10, width=1.2, pad=8)
    plt.tick_params(which='minor', length=6)
    plt.rc('lines', lw=2, dashed_pattern=[8,6], dotted_pattern=[3,6], dashdot_pattern=[10,4,3,4])
    plt.rc('legend', fontsize=18, frameon=False, handlelength=2.8, handletextpad=0.3, borderpad=1)
    plt.rc('savefig',transparent=True, bbox='tight')
    return (fig, ax)
double_dotted_pattern = [10,4,3,4,3,4]

#==== Replot Figs. 19 and 20 in Ref. [1]
m_chi_ar = np.logspace(0, np.log10(63), 100)
BR_inv_lim = 0.2
lam_SDM, sigmaSI_SDM = inv_decay_limits(model_scalarDM, BR_inv_lim, m_chi_ar)
lam_MDM, sigmaSI_MDM = inv_decay_limits(model_MajoranaDM, BR_inv_lim, m_chi_ar)
lam_VDM, sigmaSI_VDM = inv_decay_limits(model_vectorDM, BR_inv_lim, m_chi_ar)

fig, ax = setup()
plt.title(r'Invisible Higgs decays into DM,  $\mathrm{BR}_\mathrm{inv} < 20\%$')
plt.xlabel(r'$m_\chi\ (\mathrm{GeV})$')
plt.ylabel(r'$\sigma_{\chi N}^\mathrm{SI}\ (\mathrm{cm}^2)$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e0, 1e3)
ax.yaxis.set_minor_locator(ticker.LogLocator(subs=(2., 4., 6., 8.)))
plt.plot(m_chi_ar, sigmaSI_SDM, c='green', ls='-.', label='Scalar DM')
plt.plot(m_chi_ar, sigmaSI_MDM, c='red', ls='-.', label='Majorana DM')
plt.plot(m_chi_ar, sigmaSI_VDM, c='black', ls='-.', label='Vector DM')
plt.legend(loc='best')
plt.savefig('Djouadi_mchi_sigmaSI.pdf')

fig, ax = setup()
plt.title(r'Invisible Higgs decays into DM,  $\mathrm{BR}_\mathrm{inv} < 20\%$')
plt.xlabel(r'$m_\chi\ (\mathrm{GeV})$')
plt.ylabel(r'$\lambda_{H\chi\chi}$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e0, 1e3)
plt.ylim(1e-6, 1e0)
plt.plot(m_chi_ar, lam_SDM, c='green', ls='-.', label='Scalar DM')
plt.plot(m_chi_ar, lam_MDM, c='red', ls='-.', label='Majorana DM, $\Lambda = 1\ \mathrm{TeV}$')
plt.plot(m_chi_ar, lam_VDM, c='black', ls='-.', label='Vector DM')
plt.legend(loc='best')
plt.savefig('Djouadi_mchi_lam.pdf')


#===============================================
PandaX = np.loadtxt('SI_17_PandaX-II.dat')  # PandaX-II, arXiv:1708.06917
m_chi_PandaX = PandaX[:,0]
sigmaSI_PandaX = PandaX[:,1]
XENON = np.loadtxt('SI_18_XENON1T.dat')  # XENON1T, arXiv:1805.12562, PRL
m_chi_XENON = XENON[:,0]
sigmaSI_XENON = XENON[:,1]

BR_inv_ATLAS = 0.26   # ATLAS Coll., 1904.05105, PRL
BR_inv_CEPC = 2.6e-3  # Tan et al., 2001.05912, CPC
BR_inv_SPPC = 2.5e-4  # Guess for SPPC

m_chi_ar = np.append(np.logspace(0, np.log10(m_H/2 - 1), 100), np.logspace(np.log10(m_H/2 - 0.99), np.log10(m_H/2 + 0.1), 100))
lam_ATLAS, sigmaSI_ATLAS = inv_decay_limits(model_MajoranaDM, BR_inv_ATLAS, m_chi_ar)
lam_CEPC, sigmaSI_CEPC = inv_decay_limits(model_MajoranaDM, BR_inv_CEPC, m_chi_ar)
lam_SPPC, sigmaSI_SPPC = inv_decay_limits(model_MajoranaDM, BR_inv_SPPC, m_chi_ar)
fig, ax = setup()
plt.title(r'Invisible Higgs decays into Majorana DM')
plt.xlabel(r'$m_\chi\ (\mathrm{GeV})$')
plt.ylabel(r'$\sigma_{\chi N}^\mathrm{SI}\ (\mathrm{cm}^2)$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e0, 1e3)
plt.ylim(1e-50, 1e-44)
plt.plot(m_chi_PandaX, sigmaSI_PandaX, c='green', ls='-')
ax.annotate('PandaX-II', xy=(100, 2e-46), color='green', fontsize=18, rotation=19)
plt.plot(m_chi_XENON, sigmaSI_XENON, c='black', ls='-')
ax.annotate('XENON1T', xy=(130, 5e-47), color='black', fontsize=18, rotation=19)
plt.plot(m_chi_ar, sigmaSI_ATLAS, c='blue', ls='-')
ax.annotate('ATLAS', xy=(1.5, 2e-46), color='blue', fontsize=18, rotation=10)
ax.annotate('$\mathrm{BR}_\mathrm{inv} < 26\%$', xy=(1.5, 6e-47), color='blue', fontsize=18, rotation=10)
plt.plot(m_chi_ar, sigmaSI_CEPC, c='purple', ls='-.')
ax.annotate('CEPC, $\mathrm{BR}_\mathrm{inv} < 0.26\%$', xy=(1.5, 1.8e-48), color='purple', fontsize=18, rotation=9)
plt.plot(m_chi_ar, sigmaSI_SPPC, c='red', ls='-.')
ax.annotate('$\mathrm{BR}_\mathrm{inv} < 0.025\%$', xy=(2.5, 5e-50), color='red', fontsize=18, rotation=8)
plt.savefig('MajoranaDM_mchi_sigmaSI.pdf')

m_chi_ar = np.append(np.logspace(0, np.log10(m_H/2 - 1), 100), np.logspace(np.log10(m_H/2 - 0.99), np.log10(m_H/2 + 0.1), 500))
lam_ATLAS, sigmaSI_ATLAS = inv_decay_limits(model_scalarDM, BR_inv_ATLAS, m_chi_ar)
lam_CEPC, sigmaSI_CEPC = inv_decay_limits(model_scalarDM, BR_inv_CEPC, m_chi_ar)
lam_SPPC, sigmaSI_SPPC = inv_decay_limits(model_scalarDM, BR_inv_SPPC, m_chi_ar)
fig, ax = setup()
plt.title(r'Invisible Higgs decays into real scalar DM')
plt.xlabel(r'$m_\chi\ (\mathrm{GeV})$')
plt.ylabel(r'$\sigma_{\chi N}^\mathrm{SI}\ (\mathrm{cm}^2)$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e0, 1e3)
plt.ylim(1e-48, 1e-42)
plt.plot(m_chi_PandaX, sigmaSI_PandaX, c='green', ls='-')
ax.annotate('PandaX-II', xy=(100, 2e-46), color='green', fontsize=18, rotation=20)
plt.plot(m_chi_XENON, sigmaSI_XENON, c='black', ls='-')
ax.annotate('XENON1T', xy=(130, 5e-47), color='black', fontsize=18, rotation=20)
plt.plot(m_chi_ar, sigmaSI_ATLAS, c='blue', ls='-')
ax.annotate('ATLAS, $\mathrm{BR}_\mathrm{inv} < 26\%$', xy=(3, 7e-45), color='blue', fontsize=18, rotation=-32)
plt.plot(m_chi_ar, sigmaSI_CEPC, c='purple', ls='-.')
ax.annotate('CEPC, $\mathrm{BR}_\mathrm{inv} < 0.26\%$', xy=(1.5, 3.5e-47), color='purple', fontsize=18, rotation=-32)
plt.plot(m_chi_ar, sigmaSI_SPPC, c='red', ls='-.')
ax.annotate('$\mathrm{BR}_\mathrm{inv} < 0.025\%$', xy=(2, 6e-48), color='red', fontsize=18, rotation=-32)
plt.savefig('scalarDM_mchi_sigmaSI.pdf')

m_chi_ar = np.append(np.logspace(0, np.log10(m_H/2 - 1), 100), np.logspace(np.log10(m_H/2 - 0.99), np.log10(m_H/2 + 0.1), 500))
lam_ATLAS, sigmaSI_ATLAS = inv_decay_limits(model_vectorDM, BR_inv_ATLAS, m_chi_ar)
lam_CEPC, sigmaSI_CEPC = inv_decay_limits(model_vectorDM, BR_inv_CEPC, m_chi_ar)
lam_SPPC, sigmaSI_SPPC = inv_decay_limits(model_vectorDM, BR_inv_SPPC, m_chi_ar)
fig, ax = setup()
plt.title(r'Invisible Higgs decays into real vector DM')
plt.xlabel(r'$m_\chi\ (\mathrm{GeV})$')
plt.ylabel(r'$\sigma_{\chi N}^\mathrm{SI}\ (\mathrm{cm}^2)$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e0, 1e3)
plt.ylim(1e-53, 1e-44)
ax.yaxis.set_minor_locator(ticker.LogLocator(subs=(2., 4., 6., 8.)))
plt.plot(m_chi_PandaX, sigmaSI_PandaX, c='green', ls='-')
ax.annotate('PandaX-II', xy=(100, 2e-46), color='green', fontsize=18, rotation=14)
plt.plot(m_chi_XENON, sigmaSI_XENON, c='black', ls='-')
ax.annotate('XENON1T', xy=(130, 3e-47), color='black', fontsize=18, rotation=14)
plt.plot(m_chi_ar, sigmaSI_CEPC, c='purple', ls='-.')
plt.plot(m_chi_ar, sigmaSI_ATLAS, c='blue', ls='-')
ax.annotate('ATLAS, $\mathrm{BR}_\mathrm{inv} < 26\%$', xy=(1.5, 1.2e-49), color='blue', fontsize=18, rotation=30)
ax.annotate('CEPC, $\mathrm{BR}_\mathrm{inv} < 0.26\%$', xy=(2.5, 3e-51), color='purple', fontsize=18, rotation=30)
plt.plot(m_chi_ar, sigmaSI_SPPC, c='red', ls='-.')
ax.annotate('$\mathrm{BR}_\mathrm{inv} < 0.025\%$', xy=(5, 2e-52), color='red', fontsize=18, rotation=30)
plt.savefig('vectorDM_mchi_sigmaSI.pdf')




