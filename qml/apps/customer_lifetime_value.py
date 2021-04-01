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
df=pd.read_csv(DATA_PATH.joinpath("Customer Lifetime Value Online Retail.csv"),encoding="cp1252")
df=df.drop(['Description'], axis=1)
df['CustomerID'] = df['CustomerID'].astype(int)
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
            html.H1(id="forecasted-revenue-output", className="card-title"),
            html.P(
                "Cust. Lifetime Value ($ M)",
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
        dbc.NavItem(dbc.NavLink("Telco Customer Churn", active=False,href="/apps/telco_customer_churn")),
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

   html.Div(
    [


# prompts row
  dbc.Row([
    # start sidebar
    dbc.Col([

      dbc.Input(id="number-of-clv-months-input", placeholder="Enter No of Months for CLV...", type="Number", value=12, min=1, max=48,
        style={'margin-left':'3px','margin-right':'5px','margin-top':'3px'}),
            html.Br(),
      dcc.Dropdown(id='country-input',multi=True, value=df['Country'].unique()[1:11],
      options=[{'label':x,'value':x} for x in sorted(df['Country'].unique())],
      style={'margin-bottom': '7px','margin-left':'3px','margin-right':'5px'}),
    ],
    md=3,
    style={'margin-bottom': '2px','margin-top': '2px','margin-left': '0px','border-style': 'ridge','border-color': 'teal'}
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
      html.Div(id='customer-lifetime-value-output'),
      
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

@app.callback(
  Output('customer-lifetime-value-output', 'children'), 
  Output('forecasted-revenue-output', 'children'), 
  Output('customer-lifetime-value-graph-output','figure'),
  Input('number-of-clv-months-input','value'),
  )
def customer_lifetime_value(clv_months):
    t=12
    if t=='':
      t=12
    else:
      t=int(clv_months)

    last_order_date=df['Date'].max()
    lifetimes_txn_data = summary_data_from_transaction_data(df, 'CustomerID', 'Date', monetary_value_col='TotalSales', observation_period_end=last_order_date).reset_index()
    lifetimes_txn_data=lifetimes_txn_data[lifetimes_txn_data['CustomerID']!='nan']
    bgf_model=BetaGeoFitter(penalizer_coef=0.0)
    bgf_model.fit(lifetimes_txn_data['frequency'],lifetimes_txn_data['recency'],lifetimes_txn_data['T'])  
    lifetimes_txn_data['predicted_num_of_txns'] = round(bgf_model.conditional_expected_number_of_purchases_up_to_time(t, lifetimes_txn_data['frequency'], lifetimes_txn_data['recency'], lifetimes_txn_data['T']),2)
    lifetimes_txn_data=lifetimes_txn_data.sort_values(by='predicted_num_of_txns', ascending=False)
    lifetimes_txn_data.head(t)
    # Get customers with frequency >0
    lifetimes_txn_data=lifetimes_txn_data[lifetimes_txn_data['frequency']>0]
    ggf_model = GammaGammaFitter(penalizer_coef = 0)
    ggf_model.fit(lifetimes_txn_data['frequency'],lifetimes_txn_data['monetary_value'])
    lifetimes_txn_data['predicted_value_of_txn'] = round(ggf_model.conditional_expected_average_profit(
        lifetimes_txn_data['frequency'],lifetimes_txn_data['monetary_value']), 2)
    rate=0.01 # monthly discount rate ~ 12.7% annually
    lifetimes_txn_data['CLV'] = round(ggf_model.customer_lifetime_value(
        bgf_model, #the model to use to predict the number of future transactions
        lifetimes_txn_data['frequency'],
        lifetimes_txn_data['recency'],
        lifetimes_txn_data['T'],
        lifetimes_txn_data['monetary_value'],
        time=t,
        discount_rate=rate
    ), 2)
    lifetimes_txn_data.columns=['Customer No.','Frequency','Recency','Age (T)','Monetary Value','Predicted No. of Txns','Predicted Value of Txns','Customer Lifetime Value (CLV)']
    # figure
    revenue_per_customers_df=lifetimes_txn_data.groupby('Customer No.', as_index=False )['Customer Lifetime Value (CLV)'].sum().sort_values(by="Customer Lifetime Value (CLV)",ascending=False)
    revenue_per_customers_df=revenue_per_customers_df[revenue_per_customers_df['Customer No.']!='nan']
    fig=px.bar(revenue_per_customers_df.head(10),x='Customer No.',y='Customer Lifetime Value (CLV)',color='Customer No.',text='Customer Lifetime Value (CLV)',
    title='Customer Lifetime Value (CLV) for '+str(t)+' Months')
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.80),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
    return  dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in lifetimes_txn_data.columns],
                    data=lifetimes_txn_data.to_dict('records'),
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

                    ),round(lifetimes_txn_data['Customer Lifetime Value (CLV)'].sum()/1000000,2), fig


