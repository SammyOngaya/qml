# Dash dependencies import
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_uploader as du
import uuid
import pathlib
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
from dash.dependencies import Input, Output,State
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
px.defaults.template = "ggplot2"
plt.style.use('ggplot')
# End Dash dependencies import

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



import os
import io
import shutil

from app import app, server


PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("../datasets").resolve()
df=pd.read_csv(DATA_PATH.joinpath("Customer Lifetime Value Online Retail.csv"),encoding="cp1252")

def clean_data(df):
	df = df[pd.notnull(df['CustomerID'])]
	df=df[df['Quantity']>0]
	df['CustomerID'] = df['CustomerID'].astype(int)
	df['CustomerID'] = df['CustomerID'].astype(str) 
	df['Date'] = pd.to_datetime(df['InvoiceDate'], format="%d/%m/%Y %H:%M").dt.date
	df['TotalSales']=df['Quantity']*df['UnitPrice']
	df['TotalSales']=round(df['TotalSales'],2)
	return df
df=clean_data(df)


layout=dbc.Container([

   dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Telco Customer Churn", active=False,href="/apps/telco_customer_churn")),
        dbc.NavItem(dbc.NavLink("Telco Customer Survival Analysis", active=False,href="/apps/telco_customer_survival_analysis")),
        dbc.NavItem(dbc.NavLink("Customer Lifetime Value", active=False,href="/apps/customer_lifetime_value")),
        dbc.NavItem(dbc.NavLink("Customer Segmentation", active=True,href="/apps/customer_segmentation")),
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
            # html.H1(df.shape[0], className="card-title"),
            html.P(
                "Total Customers",
                className="card-text",
            ),
           ],
        style={'text-align': 'center'}
          ), color="primary", inverse=True), style={'margin-top': '30px'}, md=2),

            dbc.Col(dbc.Card(dbc.CardBody( [
            # html.H1(df[df['Churn']=='Yes']['customerID'].count(), className="card-title"),
            html.P(
                "Churned Cust",
                className="card-text",
            ),
           ],
        style={'text-align': 'center'}
          ), color="primary", inverse=True), style={'margin-top': '30px'}, md=2),

          dbc.Col(dbc.Card(dbc.CardBody( [
            # html.H1(df[df['Churn']=='No']['customerID'].count(), className="card-title"),
            html.P(
                "Remained Cust",
                className="card-text",
            ),
           ],
        style={'text-align': 'center'}
          ), color="primary", inverse=True), style={'margin-top': '30px'}, md=2),

          dbc.Col(dbc.Card(dbc.CardBody( [
            # html.H1(round(df[df['Churn']=='Yes']['TotalCharges'].sum()/1000,2), className="card-title"),
            html.P(
                "Churned Customer Rev. (K)",
                className="card-text",
            ),
           ],
        style={'text-align': 'center'}
          ), color="primary", inverse=True), style={'margin-top': '30px'}, md=3),

           dbc.Col(dbc.Card(dbc.CardBody( [
            # html.H1(round(df[df['Churn']=='No']['TotalCharges'].sum()/1000,2), className="card-title"),
            html.P(
                "Remained Customer Rev. (K)",
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
                            # figure=churn_distribution(df),
                            config={'displayModeBar': False },
                            ),
                          ] 
                        	),
                    			style={
                                'margin-top': '30px'
                                },
                        	md=3),
           #2.
                  dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='churn_by_gender',
                            # figure=churn_by_gender(df),  
                            config={'displayModeBar': False } 
                            ),
                          ] 
                          ),  
                          style={
                                'margin-top': '30px'
                                },
                          md=3),
   #3. 
                 dbc.Col(html.Div([                  
                    dcc.Graph(  
                            id='churn-by-contract', 
                            # figure=churn_by_contract(df),
                            config={'displayModeBar': False }  
                            ),
                          ] 
                          ),  
                          style={
                                'margin-top': '30px'  
                                },
                          md=6),

            ]
        ),

# 4. 
        dbc.Row(
            [ 
            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='distribution-by-revenue',
                            # figure=distribution_by_revenue(df),
                            config={'displayModeBar': False }
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=4),

            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='churn-by-monthlycharges',
                            # figure=churn_by_monthlycharges(df),
                            config={'displayModeBar': False }
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=8),
            ]
        ),



          dbc.Row(  
            [ 
            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='churn-by-citizenship',
                            # figure=churn_by_citizenship(df),
                            config={'displayModeBar': False }
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=4),

            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='tenure-charges-correlation',
                            # figure=tenure_charges_correlation(df),
                            config={'displayModeBar': False }
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=8),
            ]
        ),


         dbc.Row(
            [ 
            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='churn-by-tenure',
                            # figure=churn_by_tenure(df),
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

              dbc.Row(
            [ 
            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='churn-by-techsupport',
                            # figure=churn_by_techsupport(df),
                            config={'displayModeBar': False }
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=5),
            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='churn-by-payment_method',
                            # figure=churn_by_payment_method(df),
                            config={'displayModeBar': False }
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=7),

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
label="Customer Segmentation with RFM Model"), # Explore Data  Tab Name


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
                            # figure=feature_correlation(df),
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
                    dcc.Graph(
                            id='feature-importance',
                            # figure=feature_importance(feat_importance_df),
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

         dbc.Row(
            [ 
             dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='uac-roc',
                            # figure=uac_roc(telco_churm_metrics_df),
                            config={'displayModeBar': False }
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=6),

            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='random-forest-confusion-matrix',
                            # figure=random_forest_confusion_matrix(telco_churm_metrics_df),
                            config={'displayModeBar': False }
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


            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='logistic-regression-confusion-matrix',
                            # figure=logistic_regression_confusion_matrix(telco_churm_metrics_df),
                            config={'displayModeBar': False }
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px'
                                },
                          md=6),

             dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='svm-confusion-matrix',
                            # figure=svm_confusion_matrix(telco_churm_metrics_df),
                            config={'displayModeBar': False }
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
            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='telco-churn-model-metrics-summary',
                            # figure=telco_churn_model_metrics_summary(telco_churm_metrics_df),
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
  #End  Ml Modeling Body
label="Customer Segmentation with K-Means Clustering Model"), # Ml Modeling  Tab Name



    ]
)

	],
	fluid=True
	)

