# Dash dependencies import
import dash
import dash_core_components as dcc
import dash_html_components as html
import pathlib
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
from dash.dependencies import Input, Output,State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
px.defaults.template = "ggplot2"
# End Dash dependencies import

# Data preprocessing 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# ML Algorithm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix,roc_curve,roc_auc_score
# Save model
import os
import joblib

from app import app


PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("../datasets").resolve()
TELCO_CHURN_MODEL_DATA_PATH=PATH.joinpath("../Notebooks/Churn Models").resolve()
feat_importance_df=pd.read_csv(DATA_PATH.joinpath("feature-importance.csv"))
df=pd.read_csv(DATA_PATH.joinpath("telco-customer-churn.csv"))
telco_churm_metrics_df=pd.read_json(TELCO_CHURN_MODEL_DATA_PATH.joinpath("model_metrics.json"), orient ='split', compression = 'infer')



df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')


# Revenue distribution
def distribution_by_revenue(df):
  totalcharges_attrition_df=df.groupby( ["Churn"], as_index=False )["TotalCharges"].sum()
  totalcharges_attrition_df=totalcharges_attrition_df.sort_values(by=['TotalCharges'],ascending=True)
  totalcharges_attrition_df.columns=['Churn','Revenue']
  colors = ['crimson','skyblue']
  fig=px.bar(totalcharges_attrition_df,x='Churn',y='Revenue',color='Churn',text='Revenue',color_discrete_sequence=colors)
  fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.40),autosize=True,margin=dict(t=0,b=0,l=0,r=0))
  return fig

# churn distribution
def churn_distribution(df):
  attrition_df=df.groupby(["Churn"], as_index=False )["customerID"].count()
  colors = ['skyblue','crimson']
  fig = go.Figure(data=[go.Pie(labels=attrition_df['Churn'].tolist(), values=attrition_df['customerID'].tolist(), hole=.3)])
  fig.update_layout(showlegend=False,autosize=True,annotations=[dict(text='Attrition',  font_size=20, showarrow=False)],margin=dict(t=0,b=0,l=0,r=0),height=350,colorway=colors)
  return fig

# gender_attrition_df
def churn_by_gender(df):
  gender_attrition_df=df.groupby(["Churn","gender"], as_index=False )["customerID"].count()
  gender_attrition_df.columns=['Churn','Gender','Customers']
  colors = ['skyblue','crimson']
  fig=px.bar(gender_attrition_df,x='Gender',y='Customers',color='Churn',text='Customers',color_discrete_sequence=colors,)
  fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.46),autosize=True,margin=dict(t=0,b=0,l=0,r=0)) #use barmode='stack' when stacking,
  return fig

def churn_by_contract(df):
  contract_attrition_df=df.groupby(["Churn","Contract"], as_index=False )["customerID"].count()
  contract_base_df=df.groupby(["Contract"], as_index=False )["customerID"].count()
  contract_base_df['Churn']='Customer Base'
  contract_attrition_df=contract_attrition_df.append(contract_base_df, ignore_index = True) 
  contract_attrition_df.columns=['Churn','Contract','Customers']
  contract_attrition_df=contract_attrition_df.sort_values(by=['Contract', 'Customers'],ascending=True)
  colors = ['crimson','skyblue','teal']
  fig=px.bar(contract_attrition_df,x='Contract',y='Customers',color='Churn',text='Customers',color_discrete_sequence=colors,barmode="group")
  fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.50),autosize=True,margin=dict(t=0,b=0,l=0,r=0)) #use barmode='stack' when stacking,
  return fig

def churn_by_monthlycharges(df):
  churn_dist = df[df['Churn']=='Yes']['MonthlyCharges']
  no_churn_dist = df[df['Churn']=='No']['MonthlyCharges']
  group_labels = ['No Churn', 'Churn Customers']
  colors = ['teal','crimson']
  fig = ff.create_distplot([no_churn_dist,churn_dist], group_labels, bin_size=[1, .10],
                        curve_type='kde',  show_rug=False, colors=colors)# override default 'kde' or 'normal'
  fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.50),autosize=True,margin=dict(t=0,b=0,l=0,r=0)) #use barmode='stack' when stacking,
  return fig

def tenure_charges_correlation(df):
  df_correlation=df[['tenure','MonthlyCharges','TotalCharges']].corr()
  fig=px.imshow(df_correlation)
  fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.40),autosize=True,margin=dict(t=0,b=0,l=0,r=0))
  return fig

def churn_by_citizenship(df):
  citizenship_attrition_df=df.groupby( [ "Churn","SeniorCitizen"], as_index=False )["customerID"].count()
  citizenship_base_df=df.groupby(["SeniorCitizen"], as_index=False )["customerID"].count()
  citizenship_base_df['Churn']='Customer Base'
  citizenship_attrition_df=citizenship_attrition_df.append(citizenship_base_df, ignore_index = True) 
  citizenship_attrition_df.columns=['Churn','Citizenship','Customers']
  citizenship_attrition_df=citizenship_attrition_df.sort_values(by=['Citizenship', 'Customers'],ascending=False)
  colors = ['teal','skyblue','crimson']
  fig=px.bar(citizenship_attrition_df,x='Customers',y=['Citizenship'],color='Churn',text='Customers',orientation="h",color_discrete_sequence=colors,barmode="group")
  fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.50),autosize=True,margin=dict(t=0,b=0,l=0,r=0))
  return fig

def churn_by_tenure(df):
  tenure_attrition_df=df.groupby( [ "Churn","tenure"], as_index=False )["customerID"].count()
  tenure_attrition_df.columns=['Churn','Tenure','Customers']
  colors = ['skyblue','crimson']
  fig = px.treemap(tenure_attrition_df, path=['Churn', 'Tenure'], values='Customers',color_discrete_sequence=colors)
  fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.50),autosize=True,margin=dict(t=0,b=0,l=0,r=0)) 
  return fig

def data_summary(df):
  data_summary_df=pd.DataFrame(df.describe())
  data_summary_df.reset_index(level=0, inplace=True)
  data_summary_df=data_summary_df.drop(columns='SeniorCitizen')
  data_summary_df.columns=['Metric','Tenure','MonthlyCharges','TotalCharges']
  fig = go.Figure(data=[go.Table(header=dict(values=list(data_summary_df.columns),fill_color='paleturquoise',
                align='left'),cells=dict(values=[data_summary_df.Metric, data_summary_df.Tenure, data_summary_df.MonthlyCharges, data_summary_df.TotalCharges],
               fill_color='lavender',align='left'))])
  fig.update_layout(showlegend=False,autosize=True,margin=dict(t=0,b=0,l=0,r=0),height=350)
  return fig


def churn_by_techsupport(df):
  techsupport_attrition_df=df.groupby( [ "Churn","TechSupport"], as_index=False )["customerID"].count()
  techsupport_base_df=df.groupby(["TechSupport"], as_index=False )["customerID"].count()
  techsupport_base_df['Churn']='Customer Base'
  techsupport_attrition_df=techsupport_attrition_df.append(techsupport_base_df, ignore_index = True) 
  techsupport_attrition_df.columns=['Churn','TechSupport','Customers']
  techsupport_attrition_df=techsupport_attrition_df.sort_values(by=['TechSupport', 'Customers'],ascending=True)
  colors = ['crimson','skyblue','teal']
  fig=px.bar(techsupport_attrition_df,x='TechSupport',y='Customers',color='Churn',text='Customers',color_discrete_sequence=colors,barmode="group")
  fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.50),autosize=True,margin=dict(t=0,b=0,l=0,r=0)) #use barmode='stack' when stacking,
  return fig


####3 MODELING ####
def feature_correlation(df):
  df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].mean()) # Impute TotalCharges null values with mean TotalCharges
  df['Churn'].replace(to_replace='Yes', value=1, inplace=True)
  df['Churn'].replace(to_replace='No', value=0, inplace=True)
  df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)  # convert SeniorCitizen column to string
  data_columns=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies','Contract', 'PaperlessBilling', 'PaymentMethod','SeniorCitizen']
  df=pd.get_dummies(df,columns=data_columns)
  churn_corr_df=pd.DataFrame(df.corr()['Churn'])
  churn_corr_df.reset_index(level=0, inplace=True)
  churn_corr_df.columns=['Features','Correlation']
  churn_corr_df["Color"] = np.where(churn_corr_df["Correlation"]<0, 'negative', 'positive')
  churn_corr_df=churn_corr_df.sort_values(by=['Correlation'],ascending=False)
  colors = ['skyblue','orange']
  fig=px.bar(churn_corr_df,x='Features',y='Correlation',color='Color',text='Correlation',color_discrete_sequence=colors)
  fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.50),autosize=True,margin=dict(t=0,b=0,l=0,r=0))
  return fig


def feature_importance(feat_importance_df):
  feat_importance_df=feat_importance_df.sort_values(by=['Importance'],ascending=False)
  # feat_importance_df.columns=['Features','Importance']
  fig=px.bar(feat_importance_df,x='Features',y='Importance',text='Importance',color='Importance',height=650)
  fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01),autosize=True,margin=dict(t=0,b=0,l=0,r=0))
  return fig

def telco_churn_model_metrics_summary(telco_churm_metrics_df):
  unpivoted_metric_df=telco_churm_metrics_df[telco_churm_metrics_df['Type']=='Metric'][['Model','Accuracy','Precision','Recall','F_1_Score','AUC_Score']]
  unpivoted_metric_df=unpivoted_metric_df.melt(id_vars=['Model'], var_name='Metrics', value_name='Score').sort_values(by=['Score'],ascending=True)
  colors = ['crimson','skyblue','teal','orange']
  fig=px.bar(unpivoted_metric_df,x='Metrics',y='Score',color='Model',text='Score',color_discrete_sequence=colors,barmode="group")
  fig.update_layout(legend=dict(yanchor="top",y=0.95,xanchor="left",x=0.01),autosize=True,margin=dict(t=0,b=0,l=0,r=0)) #use barmode='stack' when stacking,
  return fig

def uac_roc(telco_churm_metrics_df):
  uac_roc_df=telco_churm_metrics_df[telco_churm_metrics_df['Type']=='ROC'][['Model','Confusion_Matrix_ROC']]
  uac_roc_df=uac_roc_df.sort_values(by=['Model'],ascending=True)
  uac_roc_df=uac_roc_df.set_index('Model').transpose()
  uac_roc_fig = go.Figure()
  uac_roc_fig.add_trace(go.Scatter(x=uac_roc_df['Logistic Regression FPR'][0], y=uac_roc_df['Logistic Regression TPR'][0],name='Logistic Regression',
                                  line = dict(color='teal', width=2),line_shape='spline'))
  uac_roc_fig.add_trace(go.Scatter(x=uac_roc_df['Random Forest FPR'][0], y=uac_roc_df['Random Forest TPR'][0],name='Random Forest',
                                  line = dict(color='royalblue', width=2),line_shape='spline'))
  uac_roc_fig.add_trace(go.Scatter(x=uac_roc_df['Support Vector Machine FPR'][0], y=uac_roc_df['Support Vector Machine TPR'][0],name='Support Vector Machine',
                                  line = dict(color='orange', width=2),line_shape='spline'))
  uac_roc_fig.add_trace(go.Scatter(x=np.array([0., 1.]), y=np.array([0., 1.]),name='Random Gues',
                                  line = dict(color='firebrick', width=4, dash='dash')))
  uac_roc_fig.update_layout(legend=dict(yanchor="bottom",y=0.05,xanchor="right",x=0.95),autosize=True,margin=dict(t=0,b=0,l=0,r=0))
  return uac_roc_fig


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
                "Churned Cust",
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
                    dcc.Graph(
                            id='churn-distribution',
                            figure=churn_distribution(df),
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
                            figure=churn_by_gender(df),  
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
                            id='churn-by-techsupport',
                            figure=churn_by_techsupport(df),
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
                            figure=distribution_by_revenue(df),
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
                            figure=churn_by_monthlycharges(df),
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
                            figure=churn_by_citizenship(df),
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
                            figure=tenure_charges_correlation(df),
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
                            figure=churn_by_tenure(df),
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
                            figure=feature_correlation(df),
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
                            figure=feature_importance(feat_importance_df),
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
                            figure=uac_roc(telco_churm_metrics_df),
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
                            id='telco-churn-model-metrics-summary',
                            figure=telco_churn_model_metrics_summary(telco_churm_metrics_df),
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





