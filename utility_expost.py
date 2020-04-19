# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:53:29 2020

@author: 330411836
"""
#%%
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nelson_siegel_svensson import NelsonSiegelCurve
from statsmodels.tsa.api import VAR
from CSModules import ALM_kit, nssmodel

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import copy
import quandl

#%%
def getfit(t1='2000-01-02', t2='2020-12-02'):
    #Load data
    df_YieldC = quandl.get(
        "USTREASURY/YIELD", authtoken="4_zrDSANo7hMt_uhyzQy")
    df_YieldC.reset_index(level=0, inplace=True)
    df_YieldC['Date'] = pd.to_datetime(df_YieldC['Date'], format="%m/%d/%Y")

    #NS Cure fit
    t_cal = df_YieldC['Date']
    i_range = np.where((t_cal > t1) & (t_cal < t2))

    t = np.array([0.08333333, 0.16666667, 0.25,
                  0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    y = np.array(df_YieldC.iloc[:, 1:]).astype(float)[i_range]
    fit_par = pd.DataFrame(np.apply_along_axis(
        lambda x: ALM_kit.ns_par(t, x), axis=1, arr=y))
    return {'df_YieldC': df_YieldC, 't_cal': t_cal.iloc[i_range], 'tact': t, 'y': y, 'fit_par': fit_par}

def stoc_simulate(getfit_data, N=5000, nlag=8):
    #Transform tau to log scale
    fit_par_VAR = getfit_data['fit_par'].iloc[:, 0:3]
    fit_par_VAR.insert(3, '3', np.log(getfit_data['fit_par'].iloc[:, 3]))

    #Stochastic VAR fitting
    model = VAR(fit_par_VAR)
    results = model.fit(nlag)

    #Extract simulated scenarios
    u_L = np.linalg.cholesky(results.resid_corr)
    u_std = np.std(results.resid, axis=0)
    u_rand = np.random.normal(size=[fit_par_VAR.shape[1], N])
    u = np.dot(u_L.conj(), u_rand)

    Var_Rand = np.dot(u.transpose(), np.diag(u_std))
    Var_Betas = results.coefs
    Var_C = results.intercept

    return {'Var_Rand': Var_Rand, 'Var_Betas': Var_Betas, 'Var_C': Var_C, 'nlag': nlag}

def forecast(getfit_data, stoc_simulate_data, itime=0, N=5000, delphi=False):
    #%% Restruct stochastic scenarios
    nlag = stoc_simulate_data['nlag']
    nTime = getfit_data['fit_par'].shape[0]
    i_par = list(range(nTime-nlag-itime, nTime-itime))
    #t1
    fit_par1 = copy.deepcopy(getfit_data['fit_par'].iloc[i_par])
    fit_par1[3] = np.log(fit_par1[3])
    #t1 + 1 day forcast
    if delphi:
        fit_par2 = np.sum([np.dot(stoc_simulate_data['Var_Betas'][i], fit_par1.iloc[-(i+1)]
                                  )+stoc_simulate_data['Var_C'] for i in range(stoc_simulate_data['nlag'])], axis=0)
        Var_fit_par = fit_par2 + stoc_simulate_data['Var_Rand']
    else:
        fit_par2 = np.array(fit_par1.iloc[-1])
        Var_fit_par = fit_par2 + stoc_simulate_data['Var_Rand']

    #Transform back
    fit_par1[3] = np.exp(fit_par1[3])
    fit_par2[3] = np.exp(fit_par2[3])
    Var_fit_par[:, 3] = np.exp(Var_fit_par[:, 3])

    #extract yield curve
    tfit = np.linspace(0, 30, 100)
    fit_par1 = fit_par1.iloc[-1]
    yfit_t1 = NelsonSiegelCurve(
        fit_par1[0], fit_par1[1], fit_par1[2], fit_par1[3])(tfit)
    yfit_t2 = NelsonSiegelCurve(
        fit_par2[0], fit_par2[1], fit_par2[2], fit_par2[3])(tfit)
    yfit_t2_stochastic = np.array([NelsonSiegelCurve(
        Var_fit_par[i][0],
        Var_fit_par[i][1],
        Var_fit_par[i][2],
        Var_fit_par[i][3])(tfit) for i in range(N)])

    return {'tfit': tfit, 'yfit_t1': yfit_t1, 'yfit_t2': yfit_t2, 'yfit_t2_stochastic': yfit_t2_stochastic,
            'Var_fit_par': Var_fit_par}

def PVCashflow_AL(forecast_data, bond_weight=[1.8, 0.2, 2.5], N=5000):

    #Asset CF generate
    cf_bond_L, t_bond_L = ALM_kit.bond_cashflow(1000, 30, 2.5, 1)
    cf_bond_M, _ = ALM_kit.bond_cashflow(1000, 10, 2, 1)
    cf_bond_S, _ = ALM_kit.bond_cashflow(1000, 2, 1, 1)
    cf_bonds = np.array([cf_bond_L,
                         np.append(cf_bond_M, np.repeat(
                             0, cf_bond_L.shape[0]-cf_bond_M.shape[0])),
                         np.append(cf_bond_S, np.repeat(0, cf_bond_L.shape[0]-cf_bond_S.shape[0]))])
    cf_weights = np.array(bond_weight)  # Bond weight (Duration matched)
    cf_singleasset = np.dot(cf_weights, cf_bonds)

    #Liabilty CF generate
    df_mort = pd.read_csv("Mx_2019.csv", sep=',')
    cf_liability, t_liability = ALM_kit.liability_cashflow(
        df_mort.loc[60:]['Total'], 3000*12)

    #PV
    LCF = [ALM_kit.PV_cashflow(cf_liability, t_liability,
                               fit_ns=forecast_data['Var_fit_par'][i]) for i in range(N)]
    ACF = [ALM_kit.PV_cashflow(cf_singleasset, t_bond_L,
                               fit_ns=forecast_data['Var_fit_par'][i]) for i in range(N)]
    ACFP = np.array(ACF)[:, 0]/np.mean(np.array(ACF)[:, 0])
    LCFP = np.array(LCF)[:, 0]/np.mean(np.array(LCF)[:, 0])
    return {'cf_asset': cf_singleasset, 'cf_liability': cf_liability, 't_asset': t_bond_L, 't_liability': t_liability, 'ACFP': ACFP, 'LCFP': LCFP}

def graph(df_forcast, df_PVCF, N=5000):
    #Initiate parameters
    N = 5000
    FR = df_PVCF['ACFP']/df_PVCF['LCFP']
    i_FR_WCS = FR.argsort()[np.int(np.ceil(0.05*N)-1)]
    i_FR_BCS = FR.argsort()[np.int(np.ceil(0.95*N)-1)]

    tfit = df_forcast['tfit']
    yfit_t2 = df_forcast['yfit_t2']
    yfit_t2_stochastic = df_forcast['yfit_t2_stochastic']
    LCFP = df_PVCF['LCFP']
    ACFP = df_PVCF['ACFP']

    # Graphing
    x = [LCFP.min(), ACFP.max()]
    f_stochastic = make_subplots(rows=2, cols=2, specs=[[{'rowspan': 2}, {}],
                                                        [None, {}]],
                                 subplot_titles=("Stochastic Yeild Curve", "PV Asset and Liability", "Funding ratio"))

    #Left Pannel: YC Graphing
    YC_BE = go.Scatter(x=tfit, y=yfit_t2, line=dict(color='black', width=1))
    YC_BE_WCS = go.Scatter(x=tfit, y=yfit_t2_stochastic[i_FR_WCS],
                           line=dict(color='Red', width=1, dash='dash'), opacity=0.5)
    YC_BE_BCS = go.Scatter(x=tfit, y=yfit_t2_stochastic[i_FR_BCS],
                           line=dict(color='Green', width=1, dash='dash'), opacity=0.5)
    YC_SimUpper = go.Scatter(x=tfit, y=yfit_t2+2*yfit_t2_stochastic.std(axis=0),
                             line=dict(color='black', width=1, dash='dash'))
    YC_SimLower = go.Scatter(x=tfit, y=yfit_t2-2*yfit_t2_stochastic.std(axis=0),
                             line=dict(color='black', width=1, dash='dash'))

    f_stochastic.add_trace(YC_BE, row=1, col=1)
    f_stochastic.add_trace(YC_BE_WCS, row=1, col=1)
    f_stochastic.add_trace(YC_BE_BCS, row=1, col=1)
    f_stochastic.add_trace(YC_SimUpper, row=1, col=1)
    f_stochastic.add_trace(YC_SimLower, row=1, col=1)
    f_stochastic.update_xaxes(title_text="Term", row=1, col=1)
    f_stochastic.update_yaxes(title_text="Rates", row=1, col=1)

    #Right top: ACPF and LCPF graphing
    AL_Corr = go.Scatter(x=LCFP, y=ACFP, mode='markers', opacity=0.5,
                         marker=dict(color='black', size=5,
                                     line=dict(width=1)))
    AL_WCS = go.Scatter(x=[LCFP[i_FR_WCS]], y=[ACFP[i_FR_WCS]], mode='markers',
                        marker=dict(color='rgba(300, 0, 0, 0.5)', size=8,
                                    line=dict(width=1)))
    AL_BCS = go.Scatter(x=[LCFP[i_FR_BCS]], y=[ACFP[i_FR_BCS]], mode='markers',
                        marker=dict(color='rgba(0, 300, 0, 0.5)', size=8,
                                    line=dict(width=1)))
    AL_MLine = go.Scatter(x=x, y=x, mode='lines',
                          line=dict(color='black', width=1))

    f_stochastic.add_trace(AL_Corr, row=1, col=2)
    f_stochastic.add_trace(AL_WCS, row=1, col=2)
    f_stochastic.add_trace(AL_BCS, row=1, col=2)

    f_stochastic.add_trace(AL_MLine, row=1, col=2)
    f_stochastic.update_xaxes(title_text="Liability", row=1, col=2)
    f_stochastic.update_yaxes(title_text="Asset", row=1, col=2)

    #Right bottom: FR Histogram
    FR_hist = go.Histogram(
        x=ACFP/LCFP, marker=dict(color='black'), opacity=0.5)
    FR_WSCLine = go.layout.Shape(
        type='line',
        x0=FR[i_FR_WCS], y0=0, x1=FR[i_FR_WCS], y1=200,
        line=dict(color='rgb(300, 0, 0)', width=3, dash='dot'))
    FR_BSCLine = go.layout.Shape(
        type='line',
        x0=FR[i_FR_BCS], y0=0, x1=FR[i_FR_BCS], y1=200,
        line=dict(color='rgb(0, 300, 0)', width=3, dash='dot'))
    f_stochastic.add_trace(FR_hist, row=2, col=2)
    f_stochastic.add_shape(FR_WSCLine, row=2, col=2)
    f_stochastic.add_shape(FR_BSCLine, row=2, col=2)
    f_stochastic.update_xaxes(title_text="Funding ratio", row=2, col=2)
    f_stochastic.update_yaxes(title_text="Counts", row=2, col=2)
    f_stochastic.update_layout(showlegend=False)

    return f_stochastic

def getFactorTables(df_forcast,df_PVCF,i_base,i_shock,N_asset = 1, N_liability = 1):
    #Factor analysis
    test_asset, test_liabiltiy, dur_asset_t2, dur_liabilities_t2 = ALM_kit.FactorAnalysis(
        df_forcast['Var_fit_par'], t1=i_base, t2=i_shock,
        cf_asset=df_PVCF['cf_asset']*N_asset, t_asset=df_PVCF['t_asset'],
        cf_liability=df_PVCF['cf_liability']*N_liability, t_liability=df_PVCF['t_liability'])

    #Reporting
    test_asset_perc = np.append(
        np.exp(np.diff(np.log(test_asset)))-1, test_asset[-1]/test_asset[0]-1)*100
    test_liabiltiy_perc = np.append(np.exp(
        np.diff(np.log(test_liabiltiy)))-1, test_liabiltiy[-1]/test_liabiltiy[0]-1)*100

    PV_asset_t1t2 = [test_asset[0], test_asset[-1]]
    PV_liabilities_t1t2 = [test_liabiltiy[0], test_liabiltiy[-1]]
    PV_ALMis = [test_asset[0]-test_liabiltiy[0],
                test_asset[-1]-test_liabiltiy[-1]]

    PV_asset_t1t2_text = [str('${:,.2f}'.format(PV_asset_t1t2[i]))
                          for i in range(len(PV_asset_t1t2))]
    PV_liabilities_t1t2_text = [str('${:,.2f}'.format(
        PV_liabilities_t1t2[i])) for i in range(len(PV_liabilities_t1t2))]
    PV_ALMis_text = [str('${:,.2f}'.format(PV_ALMis[i]))
                     for i in range(len(PV_ALMis))]

    movement_asset = [str(round(test_asset_perc[i]*100, 2)) +
                      ' bps' for i in range(len(test_asset_perc))]
    movement_liability = [str(round(test_liabiltiy_perc[i]*100, 2)) +
                          ' bps' for i in range(len(test_liabiltiy_perc))]

    #Tables
    t_sc1 = go.Table(
        header=dict(values=list(['PV', 'Asset', 'Liability', 'AL-Mismatch']),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[['Base', 'Shock'],
                           PV_asset_t1t2_text,
                           PV_liabilities_t1t2_text,
                           PV_ALMis_text],
                   fill_color='lavender',
                   align='left'))

    t_sc2 = go.Table(
        header=dict(values=list(['%Movement <br>(base to shock)', 'Asset', 'Liability']),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[['Level', 'Slope', 'Curvature', 'Tau', '<b>Total</b>'],
                           movement_asset,
                           movement_liability],
                   fill_color='lavender',
                   align='left'))

    f = make_subplots(rows=3, cols=1,
                    specs=[
                        [{"type": "table","rowspan":1}],
                        [{"type": "table","rowspan":2}],
                        [None]
                  ]
                 )

    f.add_trace(t_sc1,row=1,col=1)
    f.add_trace(t_sc2,row=2,col=1)
    return f, {"t1":t_sc1,"t2":t_sc2,'F_Asset':test_asset,'F_Liability':test_liabiltiy,'Dur_AShock':dur_asset_t2,'Dur_LShock':dur_liabilities_t2}

def getALMShock(df_PVCF,df_fitdata,N=5000,pbase=0.5,pshock=0.05):
    FR = df_PVCF['ACFP']/df_PVCF['LCFP']
    i_FR_WCS = FR.argsort()[np.int(np.ceil(pshock*N)-1)]
    i_FR_base = FR.argsort()[np.int(np.ceil(pbase*N)-1)]

    APV, _ = ALM_kit.PV_cashflow(cf=df_PVCF['cf_asset'],t=df_PVCF['t_asset'],fit_ns=df_fitdata['fit_par'].iloc[-1])
    LPV, _ = ALM_kit.PV_cashflow(cf=df_PVCF['cf_liability'],t=df_PVCF['t_liability'],fit_ns=df_fitdata['fit_par'].iloc[-1])
    n_APV = LPV/APV
    return {'i_base':i_FR_base, 'i_shock':i_FR_WCS, 'n_A':n_APV}

#%%
if __name__ == '__main__':
    df_fitdata = getfit(t1='2020-01-01',t2='2020-12-31')
    df_stochastic = stoc_simulate(df_fitdata)
    df_forcast = forecast(df_fitdata, df_stochastic)
    df_PVCF = PVCashflow_AL(df_forcast, bond_weight=[2, 1, 1])
    #f = graph(df_forcast,df_PVCF) [1.8, 0.2, 2.5]

    df_PVCF = PVCashflow_AL(df_forcast, bond_weight=[1, 1, 0])
    df_ALMShock = getALMShock(df_PVCF,pbase=0.5,pshock=0.01)
    t_sc1, t_sc2, x = getFactorTables(
            df_forcast,df_PVCF,
            df_ALMShock['i_base'],df_ALMShock['i_shock'],
            N_asset=df_ALMShock['n_A'])

    go.Figure(data=t_sc1)
    go.Figure(data=t_sc2)

    t_port = go.Table(
        header=dict(values=list(['PV', 'Asset', 'Liability', 'AL-Mismatch']),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[['Base', 'Shock'],
                        PV_asset_t1t2_text,
                        PV_liabilities_t1t2_text,
                        PV_ALMis_text],
                fill_color='lavender',
                align='left'))



# %% Exceedance probability
# ExceedProb = go.Scatter(x=-xvalues, y=Exceed_Prob)
# ExceedProb_WSCline = go.layout.Shape(x0=0, x1=-xvalues.min(), y0=5, y1=5,
#                                             type='line', line=dict(color='red', width=1, dash='dash'))

# f_stochastic.add_trace(ExceedProb, row=2, col=2)
# f_stochastic.add_shape(ExceedProb_WSCline,row=2,col=2)
# f_stochastic.update_xaxes(title_text="Amount of liability exceeding asset", row=2, col=2)
# f_stochastic.update_yaxes(title_text="Probability", row=2, col=2)

# #%%

# #%%
# plt.scatter(df_PVCF['LCFP'],df_PVCF['ACFP'])

# #%%
# df_forcast = forecast(df_fitdata,df_stochastic,itime=100,delphi=True)
# plt.plot(df_forcast['yfit_t1'],color='black')
# plt.plot(df_forcast['yfit_t2'],color='red')
# df_forcast = forecast(df_fitdata,df_stochastic,itime=99,delphi=True)
# plt.plot(df_forcast['yfit_t1'],color='grey')
# plt.plot(df_forcast['yfit_t2'],color='blue')

# #%%
# plt.plot(np.percentile(df_forcast['yfit_t2_stochastic'],5,axis=0))
# plt.plot(np.percentile(df_forcast['yfit_t2_stochastic'],50,axis=0))
# plt.plot(df_forcast['yfit_t2'])
# plt.plot(np.percentile(df_forcast['yfit_t2_stochastic'],95,axis=0))


# #%%
# FMissMatch = ACFP-LCPF
# xvalues = np.linspace(0, FMissMatch.min(), 20)
# Exceed_Prob = [stats.percentileofscore(FMissMatch, i) for i in xvalues]
