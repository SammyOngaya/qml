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
  df['Churn']=df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0 )
  df=df.dropna()
  return df
df=process_data(df)

time = df['tenure']
event= df['Churn'] 

# Train Kaplan Meier model
kmf = KaplanMeierFitter()
def train_kmf(time,event,kmf):
    kmf.fit(time, event,label='Kaplan Meier Estimate')
    return kmf
kmf_model=train_kmf(time,event,kmf)

# Kaplan Meier overrall population visualization
def kmf_survival_function(kmf):
    kmf_survival_func_df=pd.DataFrame(kmf.survival_function_).reset_index()
    kmf_confidence_df=pd.DataFrame(kmf.confidence_interval_)
    kmf_density_df=pd.DataFrame(kmf.cumulative_density_).reset_index()
    kmf_density_df.columns=['timeline','Kaplan Meier Estimate Density']
    kmf_df=pd.concat([kmf_survival_func_df,kmf_confidence_df,kmf_density_df['Kaplan Meier Estimate Density']],axis=1)

    kmf_df[['timeline', 'Kaplan Meier Estimate', 'Kaplan Meier Estimate_lower_0.95',
           'Kaplan Meier Estimate_upper_0.95', 'Kaplan Meier Estimate Density']]=round(kmf_df[['timeline', 'Kaplan Meier Estimate', 'Kaplan Meier Estimate_lower_0.95',
           'Kaplan Meier Estimate_upper_0.95', 'Kaplan Meier Estimate Density']],2)
    kmf_df['timeline'] = kmf_df['timeline'].astype(int) 
    return kmf_df
kmf_df=kmf_survival_function(kmf_model)

def plot_kmf_model_survival_function(kmf_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kmf_df['timeline'], y=kmf_df['Kaplan Meier Estimate_upper_0.95'],name='Kaplan Meier Estimate Upper Confidence (95%)',
                                    line = dict(color='rgb(0,176,246)', width=1,  dash='dash'),line_shape='linear'))
    fig.add_trace(go.Scatter(x=kmf_df['timeline'], y=kmf_df['Kaplan Meier Estimate'],name='Kaplan Meier Estimate',
                                    line = dict(color='rgb(0,176,246)', width=3, ),line_shape='vh'))
    fig.add_trace(go.Scatter(x=kmf_df['timeline'], y=kmf_df['Kaplan Meier Estimate_lower_0.95'],name='Kaplan Meier Estimate Lower Confidence (95%)',
                                    line = dict(color='rgb(0,176,246)', width=1,  dash='dash' ),line_shape='linear'))
    fig.update_layout(title={'text': 'Kaplan Meier Survival Analysis Curve for Overall Data','y':0.9,'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                              legend=dict(yanchor="bottom",y=0.80,xanchor="right",x=0.95),autosize=True,margin=dict(t=70,b=0,l=0,r=0))
    return fig

def kmf_model_cumulative_density(kmf_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kmf_df['timeline'], y=kmf_df['Kaplan Meier Estimate Density'],name='Kaplan Meier Cumulative Density',
                                    line = dict(color='orange', width=3, ),line_shape='vh'))
    fig.update_layout(title={'text': 'Kaplan Meier Survival Analysis Curve for Overall Data','y':0.9,'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                              legend=dict(yanchor="bottom",y=0.80,xanchor="right",x=0.95),autosize=True,margin=dict(t=70,b=0,l=0,r=0))
    return fig

# def compute_kmf_log_plot(kmf):  
#     kmf_log_fig = plt.figure(figsize=(9,9))
#     kmf_log_fig=kmf.plot_loglogs()
#     kmf_log_img = BytesIO()
#     kmf_log_fig.figure.savefig(kmf_log_img, format='PNG')
# #     plt.close()
#     kmf_log_img.seek(0)
# #     return 'data:image/png;base64,{}'.format(base64.b64encode(kmf_log_img.getvalue()).decode()) # Uncomment when using on plotly Dash
#     return  kmf_log_fig
  
def kmf_overral_data_table(kmf_df):
    kmf_df=kmf_df[['timeline','Kaplan Meier Estimate','Kaplan Meier Estimate Density']]
    kmf_df.columns=['timeline','KM Estimate','Density']
    kmf_data_table=dash_table.DataTable(
                    id='kmf_data_table',
                    columns=[{"name": i, "id": i} for i in kmf_df.columns],
                    data=kmf_df.to_dict('records'),
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
                    )
    return  kmf_data_table


# kmf by contract cohort
def kmf_contract_cohort(df,kmf):
    contract_cohorts = df['Contract'] 
    month_to_month_cohort = (contract_cohorts == 'Month-to-month')       
    one_year_cohort = (contract_cohorts == 'One year')  
    two_year_cohort = (contract_cohorts == 'Two year')  
    kmf.fit(time[month_to_month_cohort], event[month_to_month_cohort], label='Month-to-month')
    m_on_m_kmf_df=pd.DataFrame(kmf.survival_function_).reset_index()
    kmf.fit(time[one_year_cohort], event[one_year_cohort], label='One Year')
    one_year_kmf_df=pd.DataFrame(kmf.survival_function_).reset_index()
    kmf.fit(time[two_year_cohort], event[two_year_cohort], label='Two Year')
    two_year_kmf_df=pd.DataFrame(kmf.survival_function_).reset_index()
    contract_cohort_df=pd.concat([m_on_m_kmf_df,one_year_kmf_df['One Year'],two_year_kmf_df['Two Year']],axis=1)
    contract_cohort_df[['Month-to-month', 'One Year', 'Two Year']]=round(contract_cohort_df[['Month-to-month', 'One Year', 'Two Year']],2)
    contract_cohort_df['timeline'] = contract_cohort_df['timeline'].astype(int) 
    return contract_cohort_df
contract_cohort_df=kmf_contract_cohort(df,kmf)

def kmf_contract_cohort_fig(contract_cohort_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=contract_cohort_df['timeline'], y=contract_cohort_df['Month-to-month'],name='Month to month',
                                    line = dict(color='orange', width=3, ),line_shape='vh'))
    fig.add_trace(go.Scatter(x=contract_cohort_df['timeline'], y=contract_cohort_df['One Year'],name='One Year',
                                    line = dict(color='rgb(0,176,246)', width=3, ),line_shape='vh'))
    fig.add_trace(go.Scatter(x=contract_cohort_df['timeline'], y=contract_cohort_df['Two Year'],name='Two Year',
                                    line = dict(color='green', width=3, ),line_shape='vh'))
    fig.update_layout(title={'text': 'Kaplan Meier Survival Analysis Curve by Contract Cohorts','y':0.9,'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                              legend=dict(yanchor="bottom",y=0.05,xanchor="right",x=0.30),autosize=True,margin=dict(t=70,b=0,l=0,r=0))
    return fig


# kmf by dependent cohort
def kmf_dependent_cohort(df,kmf):
    dependents_cohorts = df['Dependents'] 
    yes_cohort = (dependents_cohorts == 'Yes')   
    no_cohort = (dependents_cohorts == 'No')  
    kmf.fit(time[yes_cohort], event[yes_cohort], label='Has Dependents')
    has_dependent_kmf_df=pd.DataFrame(kmf.survival_function_).reset_index()
    kmf.fit(time[no_cohort], event[no_cohort], label='No Dependents')
    no_dependent_kmf_df=pd.DataFrame(kmf.survival_function_).reset_index()
    dependent_cohort_df=pd.concat([has_dependent_kmf_df,no_dependent_kmf_df['No Dependents']],axis=1)
    dependent_cohort_df[['Has Dependents', 'No Dependents']]=round(dependent_cohort_df[['Has Dependents', 'No Dependents']],2)
    dependent_cohort_df['timeline'] = dependent_cohort_df['timeline'].astype(int) 
    return dependent_cohort_df

dependent_cohort_df=kmf_dependent_cohort(df,kmf)

def kmf_dependent_cohort_fig(dependent_cohort_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dependent_cohort_df['timeline'], y=dependent_cohort_df['Has Dependents'],name='Has Dependents',
                                    line = dict(color='orange', width=3, ),line_shape='vh'))
    fig.add_trace(go.Scatter(x=dependent_cohort_df['timeline'], y=dependent_cohort_df['No Dependents'],name='No Dependents',
                                    line = dict(color='rgb(0,176,246)', width=3, ),line_shape='vh'))
    fig.update_layout(title={'text': 'Kaplan Meier Survival Analysis Curve by Dependents Cohorts','y':0.9,'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                              legend=dict(yanchor="bottom",y=0.05,xanchor="right",x=0.30),autosize=True,margin=dict(t=70,b=0,l=0,r=0))
    return fig



# cph Model
def process_cph_data(df):
    df=df[['customerID','tenure','Churn','gender','Partner','Dependents','PhoneService','MonthlyCharges','SeniorCitizen','StreamingTV']]
    df=df.set_index("customerID")
    df=pd.get_dummies(df, drop_first=True)
    df['tenure'] = df['tenure'].astype(int) 
    return df
cph_df=process_cph_data(df)

# Fit cph model
cph = CoxPHFitter()
def train_cph(cph_df):
    cph.fit(cph_df, 'tenure', event_col='Churn')
    return cph
cph_model=train_cph(cph_df)


# cph model summary
def cph_model_summary(cph_model):
    model_summary_df=pd.DataFrame(cph_model.summary).reset_index()
    return model_summary_df
cph_model_summary_df=cph_model_summary(cph_model)

def cph_model_summary_data_table(cph_model_summary_df):
    cph_model_summary_df=round(cph_model_summary_df,2)
    cph_model_summary_data_table=dash_table.DataTable(
                    id='cph_model_summary',
                    columns=[{"name": i, "id": i} for i in cph_model_summary_df.columns],
                    data=cph_model_summary_df.to_dict('records'),
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
                    )
    return  cph_model_summary_data_table


wft = WeibullAFTFitter()
wft.fit(cph_df, 'tenure', event_col='Churn')


# Prediction
def process_prediction_data(cph_df,pred_customers):  
    cph_df=cph_df[cph_df.index.isin(pred_customers)]
    # cph_df=cph_df.head()
    cph_df = cph_df.iloc[0:, 2:]
    predict_df=pd.DataFrame(cph_model.predict_survival_function(cph_df)).reset_index()
    unpivoted_prediction_df=predict_df.melt(id_vars=['index'], var_name='Customers', value_name='Prediction').sort_values(by=['index'],ascending=True)
    unpivoted_prediction_df.columns=['Tenure','Customers','Prediction']
    unpivoted_prediction_df['Prediction']=round(unpivoted_prediction_df['Prediction'],2)
    unpivoted_prediction_df['Tenure'] = unpivoted_prediction_df['Tenure'].astype(int) 
    return unpivoted_prediction_df


# prediction_df=process_prediction_data(cph_df)



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
                            id='kaplan-meier-overall-survival-analysis',
                            figure=plot_kmf_model_survival_function(kmf_df),
                            config={'displayModeBar': False },
                            ),
                          ] 
                          ),
                          style={
                                # 'margin-top': '30px'
                                },
                          md=12),
                          ]
        ),     


      dbc.Row(
            [ 
                dbc.Col(
                  dcc.Graph(
                            id='kaplan-meier-overall-survival-density',
                            figure=kmf_model_cumulative_density(kmf_df),
                            config={'displayModeBar': False },
                            ),
                   style={
                                # 'margin-top': '30px'
                                },
                          md=8),
                 dbc.Col(

                   html.Div(kmf_overral_data_table(kmf_df)),
                              style={
                                # 'margin-top': '30px'
                                },
                   # dcc.Graph(
                   #          id='kaplan-meier-overall-log-logs',
                   #          # figure=compute_kmf_log_plot(kmf),
                   #          config={'displayModeBar': False },
                   #          ),
                   #  style={
                   #              'margin-top': '30px'
                   #              },
                          md=4),
            ]
        ),   


    # row 3 start
  dbc.Row([   
    dbc.Col([
           dcc.Graph(
                           id='kaplan-meier-by-contract-cohort',
                            figure=kmf_contract_cohort_fig(contract_cohort_df),
                            config={'displayModeBar': False },
                            ),
                   
                    ], 
                    style={
                                              # 'margin-top': '30px'
                                              },
                    md=6),
               dbc.Col([
               dcc.Graph(
                           id='kaplan-meier-by-dependent-cohort',
                            figure=kmf_dependent_cohort_fig(dependent_cohort_df),
                            config={'displayModeBar': False },
                            ),
                   
                    ], 
                    style={
                                              # 'margin-top': '30px'
                                              },md=6),



    ], no_gutters=True,
    style={'margin-bottom': '2px'}
    ),
  # row 3 end


#1.
        dbc.Row(
            [ 
                dbc.Col(
                  html.Div([    
                  dcc.Dropdown(id='cph-customer-input',multi=True, value=df['customerID'].unique()[1:5],
                    options=[{'label':x,'value':x} for x in sorted(df['customerID'].unique())],
                    style={'margin-bottom': '7px','margin-left':'3px','margin-right':'5px'}),

                    dcc.Graph(
                            id='cph-prediction-graph-output',
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


  dbc.Row([   
    dbc.Col([
      html.Img(id="cph-plot-src", style={'height':'100%', 'width':'100%'}),
      ], md=8),
    dbc.Col([
      html.Img(id="wft-plot-src", style={'height':'100%', 'width':'100%'}),
      ], md=4),

    ], no_gutters=True,
    style={'margin-bottom': '2px'}
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
  Output('cph-plot-src', 'src'), 
  Input('cph-plot-src', 'id'))
def compute_cph_plot(b):  
    cph_fig = plt.figure(figsize=(16,9))
    cph_fig=cph_model.plot()
    cph_img = BytesIO()
    cph_fig.figure.savefig(cph_img, format='PNG')
    plt.close()
    cph_img.seek(0)
    return 'data:image/png;base64,{}'.format(base64.b64encode(cph_img.getvalue()).decode()) # Uncomment when using on plotly Dash
    # return  cph_fig

@app.callback(
  Output('wft-plot-src', 'src'), 
  Input('wft-plot-src', 'id'))
def compute_wft_plot(b):  
    wft_fig = plt.figure(figsize=(16,9))
    wft_fig=wft.plot()
    wft_img = BytesIO()
    wft_fig.figure.savefig(wft_img, format='PNG')
    plt.close()
    wft_img.seek(0)
    return 'data:image/png;base64,{}'.format(base64.b64encode(wft_img.getvalue()).decode()) # Uncomment when using on plotly Dash
    # return  wft_fig


@app.callback(
   Output('cph-prediction-graph-output','figure'),
   Input('cph-customer-input','value'),
  )
def cph_model_per_customer(pred_customers):
    prediction_df=process_prediction_data(cph_df,pred_customers)
    fig = px.line(prediction_df, x='Tenure', y='Prediction', color='Customers',line_shape='vh')
    fig.update_layout(title={'text': 'Cox Proportional Harzard Prediction','y':0.9,'x':0.5, 'xanchor': 'center','yanchor': 'top'},
                              legend=dict(yanchor="bottom",y=0.05,xanchor="right",x=0.20),autosize=True,margin=dict(t=70,b=0,l=0,r=0))
    return fig