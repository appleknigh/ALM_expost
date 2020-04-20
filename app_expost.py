# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 07:57:02 2020

@author: 330411836
"""

#%% Packages loading
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash
import utility_expost as utility
import time
import numpy as np
from CSModules import ALM_kit
from scipy.optimize import minimize

#%% Run workers
from rq import Queue
from worker import conn
q = Queue(connection=conn)
job_getfit = q.enqueue(utility.getfit, t1='2020-01-01', t2='2020-04-08')

#%% Run job - get and fit YC
t0 = time.time()
while job_getfit.result is None:
    t1 = time.time()
    t2 = t1-t0
    time.sleep(5)
    print('waiting: {}'.format(t2))
print('Finished! Time elapse: {}'.format(t2))
df_getfit = job_getfit.result

#%% Graph generate
df_stochastic = utility.stoc_simulate(df_getfit)
df_forcast = utility.forecast(df_getfit, df_stochastic)
df_PVCF = utility.PVCashflow_AL(df_forcast, bond_weight=[1.8, 0.2, 2.5])
f = utility.graph(df_forcast, df_PVCF)

#%% Table generate
df_PVCF = utility.PVCashflow_AL(df_forcast, bond_weight=[1, 1, 1])
df_ALMShock = utility.getALMShock(df_PVCF, df_getfit, pbase=0.5, pshock=0.005)

YTMPerc_Asset = np.round(df_ALMShock['YTM_A']*100*100,2)
FRRiskPec = np.round(100*100-df_ALMShock['Risk_FR']*100*100,2)
SharpeRatio = YTMPerc_Asset/FRRiskPec

#%%
ft, x = utility.getFactorTables(
    df_forcast, df_PVCF,
    df_ALMShock['i_shock'],i_base=0,
    N_asset=df_ALMShock['n_A'])

#%%
##### Graph of stochatsic simulations - graph
Graph = dbc.Col([
    dcc.Graph(
        id='graph_YCSimulate',
        figure=f,
        style={'height': '600px'})])

#### Stochatsic simulation of YC
YC_Shock_RatioButton = dcc.RadioItems(  # Shock scenarios select - button
    id = 'Bt_YCStrategy',
    options=[
        {'label': 'Systemic', 'value': 'Sys'},
        {'label': 'Key duration', 'value': 'KDur'}
    ],
    value='Sys',
    labelStyle={'display': 'inline-block'},
    style={'margin-top': '20px'}
)

StocLab_DurationShock = dbc.Row(  # Key duration shock - input box
    [
        html.Div(
            [
                html.Div(
                    id='Dur_text', children='Shock durations'),
                html.Div(
                    dcc.Input(id='KeyDur',
                              type='text', value='',
                              disabled = True
                              )
                )
            ], style={'columnCount': 2}
        )
    ],
    style={'margin-top': '20px'}
)

#### Asset mix selection
ALM_Asset = dbc.Row(  # Asset input box - input
    [
        html.Div(
            [
                html.Div(
                    id='Asset_mix', children='Asset Mix:'),
                html.Div(
                    dcc.Input(id='asset_mix_input',
                              type='text', value='1 1 1',
                              disabled = True
                              )
                ),
            ], style={'columnCount': 2}
        ),
    ],style={'margin-top': '20px'}
)

ALM_Strategy_RatioButton = dcc.RadioItems(  # Asset strategy selection - button
    id = 'Bt_AssetStrategy',
    options=[
        {'label': 'Dur Medium', 'value': 'Dur_Center'},
        {'label': 'Dur Straddle', 'value': 'DUR_Straddle'},
        {'label': 'Dur Long', 'value': 'DUR_Long'},
        {'label': 'Custom', 'value': 'CS'}
    ],
    value='Dur_Center',
    labelStyle={'display': 'inline-block'},
    style={'margin-top': '20px'}
)

Var_Scenario = dbc.Row(  # Key duration shock - input box
    [
        html.Div(
            [
                html.Div(
                    id='VaR_text', children='VaR percentile'),
                html.Div(
                    dcc.Input(id='VaR_Perc',
                              type='number', value=0.005,
                              disabled = False
                              )
                )
            ], style={'columnCount': 2}
        )
    ],
    style={'margin-top': '20px'}
)

#%% Dash start up
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

body = dbc.Container(
    [
        dbc.Row([Graph]),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Center(
                            [html.H5(id='L0Col_Title', children='Stochastic YC simulator')],style={'margin-top': '20px'}),
                        YC_Shock_RatioButton,
                        StocLab_DurationShock,
                        html.Center([html.H5(id='L1Col_Title', children='Asset mix [Long, Medium, Short]')],style={'margin-top': '20px'}),
                        ALM_Strategy_RatioButton,
                        ALM_Asset,
                        html.Center([html.H5(id='L2Col_Title', children='VaR scenarios')],style={'margin-top': '20px'}),
                        Var_Scenario,
                        html.Button('Submit', id='sub_button',style={'margin-top': '20px','width':"100%"})
                    ], style={'border': '1px solid', 'margin': '20px'}
                ),
                dbc.Col(
                    [
                        dcc.Tabs([
                            dcc.Tab(
                                label='Portfolio', children=[
                                    html.Center(
                                        [html.H5(id='RCol_Title', children='ALM Analytics')]),
                                    html.Div(id='Return_Port',
                                             children='Return - YTM Asset: {} bps'.format(YTMPerc_Asset)),
                                    html.Div(id='Risk_Port',
                                             children='Risk - FR VaR: {} bps'.format(FRRiskPec)),
                                    html.Div(id='Sharpe', children='Sharpe: {}'.format(SharpeRatio)),
                                    html.Div(id='asset_dur', children='Duration asset: _'),
                                    html.Div(id='liability_dur', children='Duration liability: _')
                                ]
                            ),
                            dcc.Tab(id='VaR_Tab',
                                label='VaR', children=[dcc.Graph(figure=ft)]
                            )
                        ])
                    ], style={'border': '1px solid', 'margin': '20px'}
                )
            ]
        )
    ]
)

app.layout = html.Div([body])

#%%

@app.callback( #Strategy selection
    [Output(component_id='KeyDur',component_property='value'),
    Output(component_id='asset_mix_input',component_property='value'),
    Output(component_id='KeyDur',component_property='disabled'),
    Output(component_id='asset_mix_input',component_property='disabled')],
    [Input(component_id='Bt_YCStrategy', component_property='value'),
    Input(component_id='Bt_AssetStrategy', component_property='value')]
)

def func_In_YCShock(select_YCStrategy,select_AssetStrategy):
    # YC Strategy
    if select_YCStrategy == 'Sys':
        KeyDur_value = 'All'
        DurDisable_status = True
    else:
        KeyDur_value = '0.2 1 5'
        DurDisable_status = False

    # Asset mix
    def fun_loss(x):
        res_loss = utility.PVCashflow_AL(df_forcast, bond_weight=abs(x),N=1)
        return abs(res_loss['ADur']-res_loss['LDur'])
    
    def opt_dur(asset_value,bnds):
        res = minimize(fun_loss, asset_value,bounds=bnds)        
        asset_value_norm = np.round(res.x/np.sum(res.x),3)
        asset_text = ' '.join(asset_value_norm.astype(str))
        return asset_text
    bnds = ((0,1),(0,1),(0,1))

    if select_AssetStrategy == 'Dur_Center':
        asset_value = np.array([0, 0.5, 0.5])
        asset_text = opt_dur(asset_value,bnds)
        AssetDisable_status = True
    elif select_AssetStrategy == 'DUR_Straddle':
        asset_value = np.array([0.5,0,0.5])
        asset_text = opt_dur(asset_value,bnds)
        AssetDisable_status = True
    elif select_AssetStrategy == 'DUR_Long':
        asset_value = np.array([1,0,0])
        asset_text = opt_dur(asset_value,bnds)
        AssetDisable_status = True
    else:
        asset_text = '0.2 1 5'
        AssetDisable_status = False
    
    return KeyDur_value, asset_text, DurDisable_status, AssetDisable_status

@app.callback( #Asset update
     [Output(component_id='graph_YCSimulate',component_property='figure'),
     Output(component_id='VaR_Tab',component_property='children'),
     Output(component_id='Return_Port',component_property='children'),
     Output(component_id='Risk_Port',component_property='children'),
     Output(component_id='Sharpe',component_property='children'),
     Output(component_id='asset_dur',component_property='children'),
     Output(component_id='liability_dur',component_property='children')],
     [Input(component_id='sub_button',component_property='n_clicks')],
     [State(component_id='asset_mix_input',component_property='value'),
     State(component_id='VaR_Perc',component_property='value')])

def graph_update(n_clicks_duration,asset_text,VaR_perc_num):
    asset_mix = np.fromstring(asset_text, dtype=float, sep=' ') 
        
    df_PVCF = utility.PVCashflow_AL(df_forcast, bond_weight=asset_mix)

    if VaR_perc_num<1 and VaR_perc_num>0:
        df_ALMShock = utility.getALMShock(df_PVCF, df_getfit, pbase=0.5, pshock=VaR_perc_num)
        f = utility.graph(df_forcast, df_PVCF,pWCS=VaR_perc_num)
    else:
        df_ALMShock = utility.getALMShock(df_PVCF, df_getfit, pbase=0.5, pshock=0.005)
        f = utility.graph(df_forcast, df_PVCF)
        print(VaR_perc_num)
        print('error, please enter a number between 0 and 1 for VaR perc')

    ft, _ = utility.getFactorTables(
        df_forcast, df_PVCF,
        df_ALMShock['i_shock'],i_base=0,
        N_asset=df_ALMShock['n_A'])

    YTMPerc_Asset = np.round(df_ALMShock['YTM_A']*100*100,2)
    FRRiskPec = np.round(100*100-df_ALMShock['Risk_FR']*100*100,2)
    SharpeRatio = YTMPerc_Asset/FRRiskPec

    RT = 'Return - YTM Asset: {} bps'.format(YTMPerc_Asset)
    RSK = 'Risk - FR VaR: {} bps'.format(FRRiskPec)
    SP = 'Sharpe: {}'.format(SharpeRatio)

    ADur = 'Asset Duration: {}'.format(df_PVCF['ADur'])
    LDur = 'Liability Duration: {}'.format(df_PVCF['LDur'])
  
    return f, dcc.Graph(figure=ft), RT, RSK, SP, ADur, LDur



# @app.callback( #Asset mix
#     [Output()],
#     [Input()]
# )


#%% Launch
app.run_server(debug=True)

#%%
