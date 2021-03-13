# import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_uploader as du
from dash.dependencies import Input, Output,State
import pandas as pd
import pathlib
import uuid

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
               html.H6("Data Exploration") , 
                  ] 
                	),
			style={
            'margin-top': '30px'
            },
                	md=4),
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

# tabs = dbc.Tabs(
#     [
#         dbc.Tab(tab1_content, label="Tab 1"),
#         # dbc.Tab(tab2_content, label="Tab 2"),
#         dbc.Tab(
#             "This tab's content is never seen", label="Tab 3", disabled=False
#         ),
#     ]
# )


	],
	fluid=True
	)




        #     html.P("This is tab 1!", className="card-text"),
        #     dbc.Button("Click here", color="success"),
        # 