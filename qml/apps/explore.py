# import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_uploader as du
from dash.dependencies import Input, Output,State
import pandas as pd
import pathlib
import uuid
import glob
import os

from app import app, server


PATH=pathlib.Path(__file__).parent
DATA_PATH=PATH.joinpath("../datasets").resolve()
du.configure_upload(app, DATA_PATH, use_upload_id=False)



layout=dbc.Container([

   dbc.NavbarSimple(
    children=[
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

	#body
	 html.Div(
    [

  

    
    #1.
        dbc.Row(
            [
                dbc.Col(html.Div([                  
                   du.Upload( id='dash-uploader',
                        max_file_size=2,  # 1800 Mb
                        filetypes=['csv', 'zip'],
                        # upload_id=uuid.uuid1(),  # Unique session id
                        text='Drag and Drop a File Here to upload!',
                        text_completed='File Sucessfully Uploaded: ',
                           ),
                  ] 
                	),
			style={
            'margin-top': '30px'
            },
                	md=4),
   #2.
                      dbc.Col(html.Div([
                    # html.H6("# Records :") , 
                    html.Div(id='callback-output'),
                   
                    
                    ]
                  ),
      style={
            'margin-top': '30px'
            },
                  md=4),
   #3. doughnut_pie_chart_with_center
                       dbc.Col(html.Div(
              [
                # html.H6("Tweet Analysis") , 
                # html.Div(id='callback-output'),
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
     
                  ),
                  md=4),

    #5. 
                   dbc.Col(html.Div(
     
                  ),
                  md=4),

    # 6
                         dbc.Col(html.Div( 

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
)
	#end body

	],
	fluid=True
	)



@app.callback(
    Output('callback-output', 'children'),
    [Input('dash-uploader', 'isCompleted')],
    [State('dash-uploader', 'fileNames'),
     State('dash-uploader', 'upload_id')],
)
def callback_on_completion(iscompleted, filenames, upload_id):
  data=[str(x) for x in filenames]
  file=str(data).replace("['","").replace("']","")
  df=pd.read_csv(DATA_PATH.joinpath(file))
  return "# Records : "+str(df.shape[0])+" | # Attributes : "+str(df.shape[1])
