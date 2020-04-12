# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 07:57:02 2020

@author: 330411836
"""
#%% Packages
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash
import utility_expost as utility
import time

#%% Run workers
from rq import Queue
from worker import conn
q = Queue(connection=conn)

#%%
job_getfit = q.enqueue(utility.getfit, t1='2019-01-01',t2='2020-12-31')

t0 = time.time()
while job_getfit.result is None:
    t1 = time.time()
    t2 = t1-t0
    time.sleep(5)
    print('waiting: {}'.format(t2))

print('Finished! Time elapse: {}'.format(t2))
df_getfit = job_getfit.result


#%%
df_stochastic = utility.stoc_simulate(df_getfit)
df_forcast = utility.forecast(df_getfit,df_stochastic)
df_PVCF = utility.PVCashflow_AL(df_forcast,bond_weight=[1.8, 0.2, 2.5])
f = utility.graph(df_forcast,df_PVCF)

#%% Dash start up
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Link", href="#")),
        dbc.DropdownMenu(
            nav=True,
            in_navbar=True,
            label="Menu",
            children=[
                dbc.DropdownMenuItem("Entry 1"),
                dbc.DropdownMenuItem("Entry 2"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Entry 3"),
            ],
        ),
    ],
    brand="Demo",
    brand_href="#",
    sticky="top",
)

#%% Dash components

Graph = dbc.Col([
    dcc.Graph(
        id='graph_YCSimulate', 
        figure=f,
        style={'height':'600px'})])

StocLab_Timer = dcc.Slider(
            id='t_range',
            min=0,
            max=df_getfit['t_cal'].shape[0]-1,
            step=1,
            value=df_getfit['t_cal'].shape[0]-1,)

YC_Shock_RatioButton = dcc.RadioItems(
    options=[
        {'label': 'Systemic', 'value': 'Sys'},
        {'label': 'Key duration', 'value': 'KDur'}
    ],
    value='Sys',
    labelStyle={'display': 'inline-block'}
)

StocLab_DurationShock = dbc.Row(
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
            ],style={'columnCount': 2}
        )
    ]
)

ALM_Strategy_RatioButton = dcc.RadioItems(
    options=[
        {'label': 'Duration Match', 'value': 'DUR'},
        {'label': 'Custom', 'value': 'CS'}
    ],
    value='CS',
    labelStyle={'display': 'inline-block'}
)

ALM_Asset = dbc.Row(
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
    ]
)

ALM_Prob = dcc.Slider(
                    id='p_range',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0)

#%% Dash layout

body = dbc.Container(
    [
        dbc.Row([Graph]),
        dbc.Row(
            [
                dbc.Col(
                    [html.Center(
                            [html.H5(id='L0Col_Title', children='Base scenario')]),
                        html.Div(id='time_range_select',
                                 children='Time stamp: _ '),                                 
                        StocLab_Timer, #Base scenario (best estimate)
                        html.Center(
                            [html.H5(id='LCol_Title', children='YC Shocks')]),
                        YC_Shock_RatioButton, #Button to select whole curve or key duration shock
                        StocLab_DurationShock, #Shock to specific duration
                        html.Center(
                            [html.H5(id='L2Col_Title',children='Asset mix',style={'padding': 10})]
                        ),
                        ALM_Strategy_RatioButton, #Button to select strategy (custom vs duration match)
                        ALM_Asset #Assigning initial weight
                        #Button to submit
                    ], style={'border': '1px solid','margin':'20px'}
                ),
                dbc.Col(
                    [
                        html.Center(
                            [html.H5(id='RCol_Title', children='ALM Analytics')]),
                        html.Div(id='Return_Port', children='Return: _ YTM'),#Return: YTM
                        html.Div(id='Risk_Port', children='Risk: _ 5perc FR'),#Return: YTM
                        html.Div(id='Sharpe', children='Sharpe: _'),#Return: YTM
                        html.Div(
                            id='prob_range_select', children='Scenario Percentile')

                    ], style={'border': '1px solid','margin':'20px'}
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Center(
                        [
                            html.Button('Submit', id='button', style={
                                        'width': '100%', 'margin-top': '50px'})
                        ]
                    )
                )
            ]
        )
    ]
)

app.layout = html.Div([navbar, body])

#%% Call back app


#%% Launch
app.run_server(debug=True)


#x_weight = np.array([1, 1, 1])
#res_func = lambda x: ALM_kit.optimize_duration(x,cf_bonds,t_bond_L,fit_par[0],dur_liabilities)
#res = minimize(res_func, x_weight,method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
#print(np.abs(res.x))

#%% 
