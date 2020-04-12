# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:56:05 2020

@author: 330411836
"""
#%%
from nelson_siegel_svensson import NelsonSiegelCurve
import numpy as np
import random
import copy
import scipy.optimize as optimize
from CSModules import nssmodel

#%%
def ns_par(t, y):
    i = ~np.isnan(y)
    nsm = nssmodel.NSCurveFamily(False)
    nsm.estimateParam(t[i],y[i])
    par = [nsm.beta0, nsm.beta1, nsm.beta2, nsm.tau0]
    return par

def yfit_beta(tfit, fit_par):
    yfit = NelsonSiegelCurve(fit_par[0], fit_par[1], fit_par[2], fit_par[3])
    c_z = yfit(tfit)
    return c_z


def bond_cashflow(par, T, coup, freq=2):
    periods = T*freq
    coupon = coup/100.*par/freq
    cf_coupon = np.array([coupon]*periods)
    cf_bond = cf_coupon

    cf_bond[-1] = cf_coupon[-1]+par
    cf_t = np.linspace(1/freq, T, periods)
    return cf_bond, cf_t


def liability_cashflow(qx, payout):
    px = 1 - qx
    tpx = np.cumprod(px)
    cf_liability = np.array(tpx*payout)
    cf_t = np.linspace(1, len(tpx), len(tpx))
    return cf_liability, cf_t


def PV_cashflow(cf, t, fit_ns):
    ns_yieldcurve = NelsonSiegelCurve(
        fit_ns[0], fit_ns[1], fit_ns[2], fit_ns[3])
    interest = (ns_yieldcurve(t))/100
    PV_cf = [cf[i]*np.exp(-interest[i]*(i+1)) for i in range(len(interest))]
    PV = np.sum(PV_cf)
    dur = np.dot(PV_cf/PV, t)
    #y = -np.log(PV/np.sum(cf))/t[-1]
    return PV, dur


def FactorAnalysis(fit_par, t1, t2, cf_asset, t_asset, cf_liability, t_liability, direct='fwd'):
    PV_liabilities_0, dur_liabilities = PV_cashflow(
        cf_liability, t_liability, fit_ns=fit_par[t1])
    PV_asset_0, dur_asset = PV_cashflow(cf_asset, t_asset, fit_ns=fit_par[t1])

    fit_par_c = copy.deepcopy(fit_par[t1])
    PV_asset = PV_asset_0
    PV_liabilities = PV_liabilities_0
    if direct == 'fwd':
        for i in range(fit_par.shape[1]):
            fit_par_c[i] = fit_par[t2][i]
            PV_liabilities_add, dur_liabilities = PV_cashflow(
                cf_liability, t_liability, fit_ns=fit_par_c)
            PV_asset_add, dur_asset = PV_cashflow(
                cf_asset, t_asset, fit_ns=fit_par_c)

            PV_asset = np.append(PV_asset, PV_asset_add)
            PV_liabilities = np.append(PV_liabilities, PV_liabilities_add)
    return PV_asset, PV_liabilities, dur_asset, dur_liabilities

def optimize_duration(x_weights,cf,t,fit_par_t,dur_liabilities):
    cf_singleasset = np.dot(np.abs(x_weights), cf)
    PV_asset_single, dur_asset = PV_cashflow(cf_singleasset, t, fit_ns=fit_par_t)
    dur_diff = abs(dur_asset-dur_liabilities)
    return dur_diff

def bond_ytm(price, cf, dt, guess=0.05):
    ytm_func = lambda y: np.sum([cf[i]/(1+y)**(dt[i]) for i in range(len(dt))]) - price
    return optimize.newton(ytm_func, guess)