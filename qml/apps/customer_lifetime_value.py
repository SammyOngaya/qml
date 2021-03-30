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
import pathlib
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


PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("../datasets").resolve()
df=pd.read_csv(DATA_PATH.joinpath("Customer Lifetime Value Online Retail Processed.csv"),encoding="cp1252")
df=df.drop(['Description'], axis=1)
df['CustomerID'] = df['CustomerID'].astype(str) 



# card definition
number_of_customers_card = [
    dbc.CardBody(
        [
            html.H1(df.shape[0], className="card-title"),
            html.P("Total Customers",
                className="card-text",
            ),
        ],
        style={'text-align': 'center'}
    ),
]

number_of_countries_card = [
    dbc.CardBody(
        [
            html.H1(df['Country'].nunique(), className="card-title"),
            html.P(
                "Customers Countries",
                className="card-text",
            ),
        ],
        style={'text-align': 'center'}
    ),
]

total_current_revenue_card = [
    dbc.CardBody(
        [
            html.H1(round(df['TotalSales'].sum()/1000000,2), className="card-title"),
            html.P(
                "Revenue To Date ($ M)",
                className="card-text",
            ),
        ],
        style={'text-align': 'center'}
    ),
]

forecasted_revenue_card = [
    dbc.CardBody(
        [
            html.H1(round(df['TotalSales'].sum()/1000000,2), className="card-title"),
            html.P(
                "Forecasted Revenue ($ M)",
                className="card-text",
            ),
        ],
        style={'text-align': 'center'}
    ),
]
#end card definition


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


# prompts row
  dbc.Row([
    # start sidebar
    dbc.Col([

      dbc.Input(id="number-of-clv-months-input", placeholder="Enter No of Months for CLV...", type="Number", min=1, max=48,
        style={'margin-left':'3px','margin-right':'5px','margin-top':'3px'}),
            html.Br(),
      dcc.Dropdown(id='country-input',multi=True, value=df['Country'].unique()[1:11],
      options=[{'label':x,'value':x} for x in sorted(df['Country'].unique())],
      style={'margin-bottom': '7px','margin-left':'3px','margin-right':'5px'}),


        dbc.Form(
            [
                dbc.FormGroup(
                    [
                       dbc.Button("Apply to Model", id="create-analysis-input", className="mr-2", color="info")
                    ],
                    className="mr-2",
                ),
            ],
            inline=True,
            ),
    ],
    md=3,
    style={'margin-bottom': '2px','margin-top': '2px','margin-left': '0px','border-style': 'ridge','border-color': 'green'}
    ),
    # end sidebar
  dbc.Col([
    html.Div(dbc.Row([ 
      html.Div(dbc.Card(number_of_countries_card, color="info", inverse=True)),
      html.Div(dbc.Card(number_of_customers_card, color="info", inverse=True),style={'padding-left': '50px'}),
      html.Div(dbc.Card(total_current_revenue_card, color="info", inverse=True),style={'padding-left': '50px'}),
      html.Div(dbc.Card(forecasted_revenue_card, color="info", inverse=True),style={'padding-left': '50px'})
      ]),
      style={'padding-left': '20px'}
      ),
    html.Hr(),
      html.Div(
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
      ),
      
    ])
  ], no_gutters=True,
  style={'margin-bottom': '1px'}),
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
),
  #End  Customer Lifetime Value Body
label="Customer Lifetime Value"), # Customer Lifetime Value Tab Name


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




@app.callback(
  Output('customer-distribution-per-country', 'figure'), 
  Input('country-input','value'),
  )
def customer_distribution_per_country(countries):
    country_df=df[df['Country'].isin(countries)]
    customer_count_df=country_df.groupby( ["Country"], as_index=False )["CustomerID"].count().sort_values(by="CustomerID",ascending=False)
    customer_count_df.columns=['Country','Customers']
    fig=px.bar(customer_count_df.head(10),x='Country',y='Customers',text='Customers',color='Country',title='Top 10 Customers Distribution per Country')
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.8),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
    return fig


@app.callback(
  Output('revenue-distribution-per-country', 'figure'), 
  Input('country-input','value'),
  )
def revenue_distribution_per_country(countries):
    country_df=df[df['Country'].isin(countries)]
    revenue_per_country_df=country_df.groupby( ["Country"], as_index=False )["TotalSales"].sum().sort_values(by="TotalSales",ascending=False)
    revenue_per_country_df.columns=['Country','Revenue']
    revenue_per_country_df=revenue_per_country_df[revenue_per_country_df['Country']!='United Kingdom']
    fig=px.bar(revenue_per_country_df.head(10),x='Country',y='Revenue',text='Revenue',color='Country',title='Countries by Revenue')
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.80),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
    return fig
  

@app.callback(
  Output('current-monthly-revenue-output', 'figure'), 
  Input('country-input','value'),
  )
def current_monthly_revenue(countries):
    revenue_per_month_df=df.groupby('Month', as_index=False )['TotalSales'].sum().sort_values(by="Month",ascending=True)
    revenue_per_month_df.columns=['Month','Revenue']
    fig=px.bar(revenue_per_month_df,x='Month',y='Revenue',color='Revenue',text='Revenue',  title='Current Monthly Revenue Distribution')
    fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.80),autosize=True,margin=dict(t=30,b=0,l=0,r=0)) 
    return fig


@app.callback(
  Output('current-customer-revenue-output', 'figure'), 
  Input('country-input','value'),
  )
def current_customer_revenue(countries):
    revenue_per_customers_df=df.groupby('CustomerID', as_index=False )['TotalSales'].sum().sort_values(by="TotalSales",ascending=False)
    revenue_per_customers_df.columns=['Customers','Revenue']
    revenue_per_customers_df=revenue_per_customers_df[revenue_per_customers_df['Customers']!='nan']
    fig=px.bar(revenue_per_customers_df.head(10),x='Customers',y='Revenue',color='Customers',text='Revenue',
    title='Customer Revenue')
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.80),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
    return fig

@app.callback(
  Output('revenue-trend-output', 'figure'), 
  Input('country-input','value'),
  # Input('create-analysis-input','n_clicks')
  )
def revenue_trend(countries):
    revenue_trend_df=df.groupby('Date', as_index=False )['TotalSales'].sum().sort_values(by="Date",ascending=True)
    revenue_trend_df.columns=['Date','Revenue']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=revenue_trend_df['Date'], y=revenue_trend_df['Revenue'],name='Revenue',
                                    line = dict(color='teal', width=2),line_shape='spline'))
    fig.update_layout(title={'text': 'Revenue Trend','y':0.9,'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                            legend=dict(yanchor="bottom",y=0.05,xanchor="right",x=0.95),autosize=True,margin=dict(t=70,b=0,l=0,r=0))
    return fig


@app.callback(
  Output('daily-revenue-output', 'figure'), 
  Input('country-input','value'),
  # Input('create-analysis-input','n_clicks')
  )
def daily_revenue(countries):
    revenue_per_day_df=df.groupby('Day', as_index=False )['TotalSales'].sum().sort_values(by="Day",ascending=True)
    revenue_per_day_df.columns=['Day','Revenue']
    fig=px.bar(revenue_per_day_df,x='Day',y='Revenue',color='Revenue',text='Revenue',title='Revenue Dist')
    fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.80),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
    return fig