# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:53:29 2020

@author: 330411836
"""
#%%
from scipy import stats
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nelson_siegel_svensson import NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
from statsmodels.tsa.api import VAR
import ALM_kit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
import quandl

#%%
def getfit():
    #Load data
    df_YieldC = quandl.get("USTREASURY/YIELD", authtoken="4_zrDSANo7hMt_uhyzQy")
    df_YieldC.reset_index(level=0, inplace=True)
    df_YieldC['Date'] = pd.to_datetime(df_YieldC['Date'], format="%m/%d/%Y")

    #NS Cure fit
    t_cal = df_YieldC['Date']
    t = np.array([0.08333333, 0.16666667, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    y = np.array(df_YieldC.iloc[:, 1:]).astype(float)
    fit_par = pd.DataFrame(np.apply_along_axis(
        lambda x: ALM_kit.ns_par(t, x), axis=1, arr=y))
    
    return {'df_YieldC':df_YieldC, 't_cal':t_cal, 't':t, 'y':y, 'fit_par':fit_par}

def stoc_simulate(getfit_data,N=5000, nlag=8):
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

    return {'Var_Rand':Var_Rand, 'Var_Betas':Var_Betas, 'Var_C':Var_C, 'nlag':nlag}

def forecast(getfit_data,stoc_simulate_data,itime=0,N=5000,delphi=False):
    #%% Restruct stochastic scenarios
    nlag = stoc_simulate_data['nlag']
    nTime = getfit_data['fit_par'].shape[0]
    i_par = list(range(nTime-nlag-itime,nTime-itime))
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
    
    return {'tfit':tfit,'yfit_t1':yfit_t1, 'yfit_t2':yfit_t2, 'yfit_t2_stochastic':yfit_t2_stochastic,
    'Var_fit_par':Var_fit_par}

def PVCashflow_AL(forecast_data,bond_weight = [1.8, 0.2, 2.5],N=5000):
       
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
    df_mort = pd.read_csv(
        "C:/Users/Robert/Documents/GitHub/ALM_Stochastic/Mx_2019.csv", sep=',')
    cf_liability, t_liability = ALM_kit.liability_cashflow(df_mort.loc[60:]['Total'], 3000*12)

    #PV
    LCF = [ALM_kit.PV_cashflow(cf_liability, t_liability,
                           fit_ns=forecast_data['Var_fit_par'][i]) for i in range(N)]
    ACF = [ALM_kit.PV_cashflow(cf_singleasset, t_bond_L,
                            fit_ns=forecast_data['Var_fit_par'][i]) for i in range(N)]
    ACFP = np.array(ACF)[:, 0]/np.mean(np.array(ACF)[:, 0])
    LCFP = np.array(LCF)[:, 0]/np.mean(np.array(LCF)[:, 0])
    return {'cf_asset':cf_singleasset, 'cf_liability':cf_liability, 'ACFP':ACFP, 'LCFP':LCFP}


#%%
df_fitdata = getfit()
df_stochastic = stoc_simulate(df_fitdata)
df_forcast = forecast(df_fitdata,df_stochastic)

#%%
df_PVCF = PVCashflow_AL(df_forcast,bond_weight=[1.8, 0.2, 2.5])

# Initiate parameters
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
line=dict(color='black',width=1))

f_stochastic.add_trace(AL_Corr, row=1, col=2)
f_stochastic.add_trace(AL_WCS, row=1, col=2)
f_stochastic.add_trace(AL_BCS, row=1, col=2)

f_stochastic.add_trace(AL_MLine, row=1, col=2)
f_stochastic.update_xaxes(title_text="Liability", row=1, col=2)
f_stochastic.update_yaxes(title_text="Asset", row=1, col=2)

#Right bottom: FR Histogram
FR_hist = go.Histogram(x=ACFP/LCFP,marker=dict(color='black'),opacity=0.5)
FR_WSCLine = go.layout.Shape(
    type='line',
    x0=FR[i_FR_WCS], y0=0, x1=FR[i_FR_WCS], y1=200,
    line=dict(color='rgb(300, 0, 0)', width=3,dash='dot'))
FR_BSCLine = go.layout.Shape(
    type='line',
    x0=FR[i_FR_BCS], y0=0, x1=FR[i_FR_BCS], y1=200,
    line=dict(color='rgb(0, 300, 0)', width=3,dash='dot'))
f_stochastic.add_trace(FR_hist, row=2, col=2)
f_stochastic.add_shape(FR_WSCLine, row=2, col=2)
f_stochastic.add_shape(FR_BSCLine, row=2, col=2)
f_stochastic.update_xaxes(title_text="Funding ratio", row=2, col=2)
f_stochastic.update_yaxes(title_text="Counts", row=2, col=2)
f_stochastic.update_layout(showlegend=False)

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