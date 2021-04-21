# Dash dependencies import
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
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


def customer_country(df):
    customer_country_count_df=df[['Country','CustomerID']].drop_duplicates()
    customer_country_count_df=customer_country_count_df.groupby( ["Country"], as_index=False )["CustomerID"].count()
    customer_country_revenue_df=df.groupby( ["Country"], as_index=False )["TotalSales"].sum()
    customer_country_revenue_df.columns=['Country','TotalSales']
    customer_country_revenue_df=round(customer_country_revenue_df,2)
    customer_country_df=pd.concat([customer_country_count_df,customer_country_revenue_df],axis=1)
    # Merge two dataframes
    customer_country_df=pd.merge(customer_country_count_df,customer_country_revenue_df, on="Country")
    # Get Country Code from a nother dataset gapminder
    country_iso_df = px.data.gapminder()
    country_iso_df=country_iso_df[['country','iso_alpha']]
    country_iso_df=country_iso_df.drop_duplicates(subset=['country'])
    country_iso_df.columns=['Country','Country Code']
    #Merge the two datasets
    customer_country_df=pd.merge(customer_country_df,country_iso_df, on="Country")
    customer_country_df['CustomerID'] = customer_country_df['CustomerID'].astype(str) 
    return customer_country_df

customer_country_df=customer_country(df)

def customer_geosegmentation(customer_country_df):
    customer_country_df=customer_country_df[customer_country_df['Country']!='United Kingdom'] # remove UK since it's a dominant country in the dataset
    fig = go.Figure(data=go.Choropleth(
        locations = customer_country_df['Country Code'],
        z = customer_country_df['TotalSales'],
        text = "Customers : "+customer_country_df['CustomerID']+"<br>Country : "+customer_country_df['Country'],
        colorscale = 'Blues',
        autocolorscale=True,
        reversescale=True,
        marker_line_color='darkgray',
        marker_line_width=0.01,
        colorbar_tickprefix = '$',
        colorbar_title = 'Total<br>Revenue US$',

    ))

    fig.update_layout(
        title_text='Geographical Segmentation of Customers with Revenue & No. of Customers',
        geo = dict(
            showlakes=True, 
            lakecolor='skyblue',
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.8),autosize=True,margin=dict(t=30,b=0,l=0,r=0)
    )
    return fig

## RFM

def recency_score (quantiles):
    if quantiles <= 17:
        return 1
    elif quantiles <= 50:
        return 2
    elif quantiles <= 141.5:
        return 3
    else:
        return 4

def frequency_score (quantiles):
    if quantiles <= 17:
        return 1
    elif quantiles <= 41:
        return 2
    elif quantiles <= 100:
        return 3
    else:
        return 4

def monetary_score (quantiles):
    if quantiles <= 307.245:
        return 1
    elif quantiles <= 674.450:
        return 2
    elif quantiles <= 1661.640:
        return 3
    else:
        return 4

def rfm_model(df):
	last_order_date=df['Date'].max()
	rfm = df.groupby('CustomerID').agg({'Date': lambda x: (last_order_date - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalSales': lambda x: x.sum()}).reset_index()
	rfm.rename(columns={'Date': 'Recency','InvoiceNo': 'Frequency','TotalSales': 'Monetary'}, inplace=True)
	quantiles = rfm.quantile(q=[0.25,0.5,0.75])
	rfm['R'] = rfm['Recency'].apply(recency_score )
	rfm['F'] = rfm['Frequency'].apply(frequency_score)
	rfm['M'] = rfm['Monetary'].apply(monetary_score)
	rfm['rfm_group'] = rfm.R.map(str) + rfm.F.map(str) + rfm.M.map(str)
	rfm['rfm_score'] = rfm[['R', 'F', 'M']].sum(axis=1)
	rfm['rfm_catrgory'] = pd.qcut(rfm.rfm_score, q = 4, labels = ['Platinum', 'Gold', 'Silver', 'Bronze']).values
	return rfm

rfm=rfm_model(df)

def rfm_customer_segments(rfm):
    rfm_category_df=rfm.groupby(["rfm_catrgory"], as_index=False )["CustomerID"].count()
    rfm_category_df=rfm_category_df.sort_values(by=['CustomerID'],ascending=False)
    rfm_category_df.columns=['Clusters','No. Customers']
    colors=['gold','orange','grey','brown']
    fig=px.bar(rfm_category_df,x='Clusters',y='No. Customers',text='No. Customers',color='Clusters',
               color_discrete_sequence=colors,title='RFM Customer Segments')
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.8),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
    return fig


def plot_rfm_clusters(rfm):
    colors=['grey','gold','orange','brown']
    fig = px.scatter(rfm, x="Recency", y="Frequency", color="rfm_catrgory",color_discrete_sequence=colors,
                     hover_data=['rfm_catrgory'], log_y=True,title="Customer Clusters Using RFM Method")
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.8),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
    return fig


def plot_3d_rfm_clusters(rfm):
    colors=['grey','gold','orange','brown']
    fig = px.scatter_3d(rfm, x="Recency", y="Frequency", z='rfm_catrgory',color_discrete_sequence=colors,
                        color='rfm_catrgory', log_y=True, title="RFM Customer Clusters", height=550)
    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.8),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
    return fig
    

def rfm_data_table(rfm):
	rfm=rfm[['CustomerID','Recency','Frequency','Monetary','rfm_score','rfm_catrgory']]
	rfm.columns=['Customer','R','F','M','Score','Cluster']
	rfm['M']=round(rfm['M'],2)
	rfm_table=dash_table.DataTable(
                    id='rfm_data_table',
                    columns=[{"name": i, "id": i} for i in rfm.columns],
                    data=rfm.to_dict('records'),
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
                    page_size= 15,
                    )
	return  rfm_table


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

            dbc.Col(
            html.H5("Select Country"),
            style={'margin-top': '15px'}, md=2),

            dbc.Col(
				      dcc.Dropdown(id='country-input',multi=True, value=df['Country'].unique()[1:11],
				      options=[{'label':x,'value':x} for x in sorted(df['Country'].unique())],
				      style={'margin-top': '15px'}),

            	 md=10),
            ]
        ),


    #1.
        dbc.Row(
            [ 
                dbc.Col(html.Div([ 
                 dcc.Graph(
                            id='customer-segmentation-per-country',
                            figure={},
                            config={'displayModeBar': False },
                            ),               
                          ] 
                        	),
                    			style={
                                'margin-top': '30px'
                                },
                        	md=6),
           #2.
                  dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='customer-revenue-segmentation-per-country',
                            figure={},  
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
                            id='customer-geodist',
                            figure=customer_geosegmentation(customer_country_df),  
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
                            id='rfm-customer-segments',
                            figure=rfm_customer_segments(rfm),
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
                            id='churn-by-monthlycharges',
                            figure=plot_rfm_clusters(rfm),
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
            dbc.Col(
            		html.Div(rfm_data_table(rfm)),
                          style={
                                'margin-top': '30px'
                                },
                          md=4),

            dbc.Col(html.Div([                  
                    dcc.Graph(
                            id='plot-3d-rfm-clusters',
                            figure=plot_3d_rfm_clusters(rfm),
                            config={'displayModeBar': False }
                            ),
                          ] 
                          ),
                          style={
                                'margin-top': '30px',
                                'height':'550px'
                                },
                          md=8),
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
                dbc.Col(

                	html.Div("@galaxydataanalytics "),
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



@app.callback(
  Output('customer-segmentation-per-country', 'figure'), 
  Input('country-input','value'),
  )
def customer_distribution_per_country(countries):
	country_df=df[df['Country'].isin(countries)]
	country_df=country_df[['Country','CustomerID']].drop_duplicates()
	customer_count_df=country_df.groupby( ["Country"], as_index=False )["CustomerID"].count().sort_values(by="CustomerID",ascending=False)
	customer_count_df.columns=['Country','Customers']
	fig=px.bar(customer_count_df.head(10),x='Country',y='Customers',text='Customers',color='Country',title='Customers Distribution per Top 10 Countries')
	fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.7),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
	return fig

@app.callback(
  Output('customer-revenue-segmentation-per-country', 'figure'), 
  Input('country-input','value'),
  )
def renevue_dist_by_country(countries):
	country_df=df[df['Country'].isin(countries)]
	revenue_per_country_df=country_df.groupby( ["Country"], as_index=False )["TotalSales"].sum().sort_values(by="TotalSales",ascending=False)
	revenue_per_country_df.columns=['Country','TotalSales']
	revenue_per_country_df=round(revenue_per_country_df,2)
	fig=px.bar(revenue_per_country_df.head(10),x='Country',y='TotalSales',text='TotalSales',color='Country',title='Revenue Distribution per Top 10 Countries')
	fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.7),autosize=True,margin=dict(t=30,b=0,l=0,r=0))
	return fig