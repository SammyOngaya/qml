# Dash dependencies import
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output,State
import dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
px.defaults.template = "ggplot2"
plt.style.use('ggplot')
import pathlib
# End Dash dependencies import

# Lifelimes libraries
from lifelines import KaplanMeierFitter,CoxPHFitter, WeibullAFTFitter
from io import BytesIO
import base64
# End Lifelimes libraries


from app import app, server


PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("../datasets").resolve()
df=pd.read_csv(DATA_PATH.joinpath("telco-customer-churn.csv"))


def process_data(df):
  df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')
  df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)  
  df=df.dropna()
  df['Churn']=df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0 )
  return df

df=process_data(df)


layout=dbc.Container([

   dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Telco Customer Churn", active=False,href="/apps/telco_customer_churn")),
        dbc.NavItem(dbc.NavLink("Telco Customer Survival Analysis", active=True,href="/apps/telco_customer_survival_analysis")),
        dbc.NavItem(dbc.NavLink("Customer Lifetime Value", active=False,href="/apps/customer_lifetime_value")),
        # dbc.NavItem(dbc.NavLink("Explore", active=True,href="/apps/explore")),
        # dbc.NavItem(dbc.NavLink("Analyse", active=True,href="#")),
        # dbc.NavItem(dbc.NavLink("Model", active=True, href="#"))
    ], 
    brand="Qaml",
    brand_href="/apps/home",
    color="primary",
    dark=True,
    style={'margin-bottom': '2px'},


),#end navigation

   html.Div(
    [

html.Hr(),
# row 2 start

   #1.
        dbc.Row(
            [ 
                dbc.Col(
                  html.Div([                  
                    dcc.Graph(
                            id='current-customer-revenue-output',
                            figure={},
                            config={'displayModeBar': False },
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=6),
   #3. 
                 dbc.Col(
                  html.Div([                  
                    dcc.Graph(
                            id='customer-lifetime-value-graph-output',
                            figure={},
                            config={'displayModeBar': False },
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=6),
            ]
        ),     


      dbc.Row(
            [ 
                dbc.Col(
               html.Img(src=app.get_asset_url('customer_lifetime_value/frequency_recency_matrix.png'), style={'height':'100%', 'width':'100%'}),
                          style={
                                'margin-top': '30px'
                                },
                          md=6),
                 dbc.Col(
                  html.Img(src=app.get_asset_url('customer_lifetime_value/probability_alive_matrix.png'), style={'height':'100%', 'width':'100%'}),
                    style={
                                'margin-top': '30px'
                                },
                          md=6),
            ]
        ),   


    # row 3 start
  dbc.Row([   
    dbc.Col([
      html.Img(id="probability-alive-src", style={'height':'100%', 'width':'100%'}),
      ], md=8),
    dbc.Col([
      html.Img(id="model-evaluation-src", style={'height':'100%', 'width':'100%'}),
      ], md=4),

    ], no_gutters=True,
    style={'margin-bottom': '2px'}
    ),
  # row 3 end


#1.
        dbc.Row(
            [ 
                dbc.Col(
                  html.Div([                  
                    dcc.Graph(
                            id='revenue-trend-output',
                            figure={},
                            config={'displayModeBar': False },
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=12),

            ]
        ),  


 #1.
        dbc.Row(
            [ 
                dbc.Col(
                  html.Div([                  
                    dcc.Graph(
                            id='customer-distribution-per-country',
                            figure={},
                            config={'displayModeBar': False },
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=6),
   #3. 
                 dbc.Col(
                  html.Div([                  
                    dcc.Graph(
                            id='revenue-distribution-per-country',
                            figure={},
                            config={'displayModeBar': False },
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=6),

            ]
        ),     


   #1.
        dbc.Row(
            [ 
                dbc.Col(
                  html.Div([                  
                    dcc.Graph(
                            id='current-monthly-revenue-output',
                            figure={},
                            config={'displayModeBar': False },
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=6),
   #3. 
                 dbc.Col(
                  html.Div([                  
                    dcc.Graph(
                            id='daily-revenue-output',
                            figure={},
                            config={'displayModeBar': False },
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=6),

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
)

  ],
  fluid=True
  )




# @app.callback(
#   Output('customer-distribution-per-country', 'figure'), 
#   Input('country-input','value'),
#   )
# def customer_distribution_per_country(countries):
#     country_df=df[df['Country'].isin(countries)]
#     customer_count_df=country_df.groupby( ["Country"], as_index=False )["CustomerID"].count().sort_values(by="CustomerID",ascending=False)
#     customer_count_df.columns=['Country','Customers']
#     fig=px.bar(customer_count_df.head(10),x='Country',y='Customers',text='Customers',color='Country',title='Top 10 Customers Distribution per Country')
#     fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.8),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
#     return fig
