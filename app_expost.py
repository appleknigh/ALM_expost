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

#%% Run workers
from rq import Queue
from worker import conn
q = Queue(connection=conn)
job_getfit = q.enqueue(utility.getfit, t1='2020-01-01', t2='2020-12-31')

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
df_PVCF = utility.PVCashflow_AL(df_forcast, bond_weight=[1, 1, 0])
df_ALMShock = utility.getALMShock(df_PVCF, df_getfit, pbase=0.5, pshock=0.005)
ft, x = utility.getFactorTables(
    df_forcast, df_PVCF,
    df_ALMShock['i_base'], df_ALMShock['i_shock'],
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
                              type='text', value=''
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
                    id='Asset_mix', children='Bond Mix: [Long, Median, Short]'),
                html.Div(
                    dcc.Input(id='asset_mix_input',
                              type='text', value='1 1 1'
                              )
                ),
            ], style={'columnCount': 2}
        ),
        #html.Button('Duration match', id='Op'),
        html.Div(id='asset_dur', children=''),
        html.Div(id='liability_dur', children='')
    ],style={'margin-top': '20px'}
)

ALM_Strategy_RatioButton = dcc.RadioItems(  # Asset strategy selection - button
    options=[
        {'label': 'Duration Match', 'value': 'DUR'},
        {'label': 'Custom', 'value': 'CS'}
    ],
    value='CS',
    labelStyle={'display': 'inline-block'},
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
                        html.Center([html.H5(id='L1Col_Title', children='Asset mix')],style={'margin-top': '20px'}),
                        ALM_Strategy_RatioButton,
                        ALM_Asset,
                        html.Button('Submit', id='button',style={'margin-top': '20px','width':"100%"})
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
                                             children='Return: _ YTM'),
                                    html.Div(id='Risk_Port',
                                             children='Risk: _ 5perc FR'),
                                    html.Div(id='Sharpe', children='Sharpe: _')
                                ]
                            ),
                            dcc.Tab(
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

#%% Launch
app.run_server(debug=True)

#%%
