# Dash dependencies import
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_uploader as du
from dash.dependencies import Input, Output,State
import pathlib
import uuid
import pandas as pd
import numpy as np
# px.defaults.template = "ggplot2"
# End Dash dependencies import

# # Data preprocessing 
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# # ML Algorithm
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# # Model evaluation
# from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix,roc_curve,roc_auc_score
# # Save model
# import os
# import joblib


from app import app, server


PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("../datasets").resolve()
df=pd.read_csv(DATA_PATH.joinpath("telco-customer-churn.csv"))

df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')



layout=dbc.Container([

   dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Telco Customer Churn", active=True,href="/apps/telco_customer_churn")),
        dbc.NavItem(dbc.NavLink("Explore", active=True,href="/apps/explore")),
        dbc.NavItem(dbc.NavLink("Clean", active=True,href="#")),
        dbc.NavItem(dbc.NavLink("Analyse", active=True,href="#")),
        dbc.NavItem(dbc.NavLink("Model", active=True, href="#"))
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
            html.H1(df.shape[0], className="card-title"),
            html.P(
                "Total Customers",
                className="card-text",
            ),
           ],
        style={'text-align': 'center'}
          ), color="primary", inverse=True), style={'margin-top': '30px'}, md=2),

            dbc.Col(dbc.Card(dbc.CardBody( [
            html.H1(df[df['Churn']=='Yes']['customerID'].count(), className="card-title"),
            html.P(
                "Churned Customers",
                className="card-text",
            ),
           ],
        style={'text-align': 'center'}
          ), color="primary", inverse=True), style={'margin-top': '30px'}, md=2),

          dbc.Col(dbc.Card(dbc.CardBody( [
            html.H1(df[df['Churn']=='No']['customerID'].count(), className="card-title"),
            html.P(
                "Remained Cust",
                className="card-text",
            ),
           ],
        style={'text-align': 'center'}
          ), color="primary", inverse=True), style={'margin-top': '30px'}, md=2),

          dbc.Col(dbc.Card(dbc.CardBody( [
            html.H1(round(df[df['Churn']=='Yes']['TotalCharges'].sum()/1000,2), className="card-title"),
            html.P(
                "Churned Customer Rev. (K)",
                className="card-text",
            ),
           ],
        style={'text-align': 'center'}
          ), color="primary", inverse=True), style={'margin-top': '30px'}, md=3),

           dbc.Col(dbc.Card(dbc.CardBody( [
            html.H1(round(df[df['Churn']=='No']['TotalCharges'].sum()/1000,2), className="card-title"),
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
               html.H6("Churn Distribution") , 
               dcc.Graph(id='churn_distribution',figure={}),
                  ] 
                	),
			style={
            'margin-top': '30px'
            },
                	md=6),
   #2.
                      dbc.Col(html.Div([
                    html.H6("Data Exploration") , 
                   
                    
                    ]
                  ),
      style={
            'margin-top': '30px',
            'font-size':'20px'
            },
                  md=4),
   #3. 
                       dbc.Col(html.Div(
              [
                html.H6("Data Exploration") , 
                   ]
                  ),
      style={
            'margin-top': '30px'
            },
                  md=4),

            ]
        ),

# 4. 
        dbc.Row(
            [
                        dbc.Col(html.Div(
     html.H6("Data Exploration") , 
                  ),
                  md=4),

    #5. 
                   dbc.Col(html.Div(
     html.H6("Data Exploration") , 
                  ),
                  md=4),

    # 6
                         dbc.Col(html.Div( 
html.H6("Data Exploration") , 
                  ),
                  md=4),
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
               html.H6("Ml Modeling") , 
                  ] 
                  ),
      style={
            'margin-top': '30px'
            },
                  md=4),
   #2.
                      dbc.Col(html.Div([
                    html.H6("Ml Modeling") , 
                   
                    
                    ]
                  ),
      style={
            'margin-top': '30px',
            'font-size':'20px'
            },
                  md=4),
   #3. 
                       dbc.Col(html.Div(
              [
                html.H6("Ml Modeling") , 
                   ]
                  ),
      style={
            'margin-top': '30px'
            },
                  md=4),

            ]
        ),

# 4. 
        dbc.Row(
            [
                        dbc.Col(html.Div(
     html.H6("Ml Modeling") , 
                  ),
                  md=4),

    #5. 
                   dbc.Col(html.Div(
     html.H6("Ml Modeling") , 
                  ),
                  md=4),

    # 6
                         dbc.Col(html.Div( 
html.H6("Ml Modeling") , 
                  ),
                  md=4),
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
label="Ml Modeling"), # Ml Modeling  Tab Name


# Ml Prediction Tab
dbc.Tab(
  # Ml Prediction Body
   html.Div(
    [
    #1.
        dbc.Row(
            [
                dbc.Col(html.Div([                  
               html.H6("Ml Prediction") , 
                  ] 
                  ),
      style={
            'margin-top': '30px'
            },
                  md=4),
   #2.
                      dbc.Col(html.Div([
                    html.H6("Ml Prediction") , 
                   
                    
                    ]
                  ),
      style={
            'margin-top': '30px',
            'font-size':'20px'
            },
                  md=4),
   #3. 
                       dbc.Col(html.Div(
              [
                html.H6("Ml Prediction") , 
                   ]
                  ),
      style={
            'margin-top': '30px'
            },
                  md=4),

            ]
        ),

# 4. 
        dbc.Row(
            [
                        dbc.Col(html.Div(
     html.H6("Ml Prediction") , 
                  ),
                  md=4),

    #5. 
                   dbc.Col(html.Div(
     html.H6("Ml Prediction") , 
                  ),
                  md=4),

    # 6
                         dbc.Col(html.Div( 
html.H6("Ml Prediction") , 
                  ),
                  md=4),
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
  #End  Ml Prediction Body
label="Ml Prediction"), # Ml Prediction  Tab Name


    ]
)

	],
	fluid=True
	)





## Data Exploration Callbacks
@app.callback(
Output('churn_distribution' , 'figure')
)
def churn_distribution():
  attrition_df=df.groupby( [ "Churn"], as_index=False )["customerID"].count()
  colors = ['skyblue','crimson']
  doughnut_attrition = go.Figure(data=[go.Pie(labels=attrition_df['Churn'].tolist(), values=attrition_df['customerID'].tolist(), hole=.3)])
  doughnut_attrition.update_layout(showlegend=False,autosize=True,annotations=[dict(text='Attrition',  font_size=20, showarrow=False)],margin=dict(t=0,b=0,l=0,r=0),height=350,colorway=colors)
  return doughnut_attrition