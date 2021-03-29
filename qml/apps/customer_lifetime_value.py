# Dash dependencies import
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_uploader as du
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
from dash.dependencies import Input, Output,State
import dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
px.defaults.template = "ggplot2"
# End Dash dependencies import

# Lifetimes libraries
from lifetimes.utils import *
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.plotting import plot_period_transactions
from lifetimes.plotting import plot_history_alive
# End Lifetimes libraries


from app import app, server


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')


layout=dbc.Container([

   dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Telco Customer Churn", active=True,href="/apps/telco_customer_churn")),
        dbc.NavItem(dbc.NavLink("Customer Lifetime Value", active=True,href="/apps/customer_lifetime_value")),
        # dbc.NavItem(dbc.NavLink("Explore", active=True,href="/apps/explore")),
        # dbc.NavItem(dbc.NavLink("Clean", active=True,href="#")),
        # dbc.NavItem(dbc.NavLink("Analyse", active=True,href="#")),
        # dbc.NavItem(dbc.NavLink("Model", active=True, href="#"))
    ], 
    brand="Qaml",
    brand_href="/apps/home",
    color="primary",
    dark=True,
    style={'margin-bottom': '2px'},


),#end navigation


dbc.Tabs(
    [

# Explore Data Tab
dbc.Tab(
  # Explore Data Body
   html.Div(
    [


#Cards Row.
        dbc.Row(
            [
          dbc.Col(dbc.Card(dbc.CardBody( [
            html.H1("Five", className="card-title"),
            html.P(
                "Churned Customer Rev. (K)",
                className="card-text",
            ),
           ],
        style={'text-align': 'center'}
          ), color="primary", inverse=True), style={'margin-top': '30px'}, md=3),
            ]
        ),


    #1.
        dbc.Row(
            [ 
                dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='churn-distribution',
                            figure={},
                            config={'displayModeBar': False },
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=3),
   
   #3. 
                 dbc.Col(
                       dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict('records'),

                    editable=True,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    column_selectable="single",
                    row_selectable="multi",
                    row_deletable=True,
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current= 0,
                    page_size= 10,

                    ),
                          style={
                                'margin-top': '30px'  
                                },
                          md=9),

            ]
        ),     

       
        # footer
    dbc.Row(
            [
                dbc.Col(html.Div("@galaxydataanalytics "),
                  style={
            'margin-top': '2px',
            'text-align':'center',
            'backgroundColor': 'rgba(120,120,120,0.2)'
            },
                 md=12)
            ]
        ),
        #end footer
    ],
        style={
            'padding-left': '3px',
            'padding-right': '3px'
            },
),
  #End  Explore Data Body
label="Explore Data"), # Explore Data  Tab Name


# Ml Modeling Tab
dbc.Tab(
  # Ml Modeling Body
   html.Div(
    [
    #1.
        dbc.Row(
            [ 
            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='feature-correlation',
                            figure={},
                            config={'displayModeBar': False }
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=12),

            ]
        ),

# 4. 
       dbc.Row(
            [ 
            dbc.Col(html.Div([                  
                    html.H1("Tab 2"),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=12),

            ]
        ),

       

        # footer
    dbc.Row(
            [
                dbc.Col(html.Div("@galaxydataanalytics "),
                  style={
            'margin-top': '2px',
            'text-align':'center',
            'backgroundColor': 'rgba(120,120,120,0.2)'
            },
                 md=12),

                 dbc.Col(
                 # Hidden div inside the app that stores the intermediate value
              html.Div(id='global-dataframe'),
          # , style={'display': 'none'}
                  style={'display': 'none'},
                 md=0),
            ]
        ),
        #end footer



     
    ],
        style={
            'padding-left': '3px',
            'padding-right': '3px'
            },
),
  #End  Ml Modeling Body
label="Ml Modeling"), # Ml Modeling  Tab Name




    ]
)

  ],
  fluid=True
  )


