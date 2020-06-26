import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import Input, Output,State
import datetime
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
import base64

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
app.config.suppress_callback_exceptions = True
app.title = "Arckifejezés felismerés"
app.layout = html.Div(
[
    (html.Div([
        html.Center(html.H1('Arckifejezés felismerés')),
        html.Hr(),
        html.Center(html.H3('Arcok felismerése egy képen, illetve a képen látható arckifejezés felismerése.')),
        html.Br(),
        html.Center(html.P('A weboldal célja két neurális modell összehasonlítása, valamint annak lehetősége, hogy a felhasználó tudja tesztelni a modelleket.')),
        html.Br(),
        html.Br(),
    ])), 
    dbc.Row(
        [
            
            dbc.Col([
            html.Center(html.H4("Ashadullah Shawon megvalósítása")),
            html.Center(html.A("Az osztályozó leírása itt érhető el", href='https://www.kaggle.com/shawon10/facial-expression-detection-cnn', style = {'color':'gray', 'font-weight':'bold'})),
            html.Br(),
            html.Center(html.P("Ebben a részben látható lesz az első modell szerinti veszteség ábrázolása, a Tanítási veszteság, ileltve a Validációs veszteség alapján.")),
            html.Br(),
            html.Center(dcc.Graph(
                id='veszteseg-graph',
                figure={
                    'data': [
                        {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'y': [1.6698, 1.3242, 1.1302, 1.0033, 0.8746, 0.7291, 0.5656, 0.4108, 0.2881, 0.2052, 0.1676, 0.1285, 0.1086, 0.1115, 0.104, 0.0952, 0.0935, 0.0794, 0.0843, 0.0724], 'type': 'scatter', 'name': 'Tanítási veszteség'},
                        {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'y': [1.7831, 1.3562, 1.1762, 1.148, 1.0692, 1.1269, 1.2335, 1.2776, 1.5797, 1.5014, 1.7935, 1.8073, 1.8007, 1.9739, 1.9873, 1.9692, 2.0034, 2.1322, 2.0265, 2.2204], 'type': 'scatter', 'name': 'Validációs veszteség'},
                    ],
                    'layout': {
                        'title': 'Veszteség ábrázolása',
                        'xaxis': dict(
                            title = 'Epoch',
                            titlefont = dict(
                                family = 'Helvetica, monospace',
                                size = 12,
                                color = '#7f7f7f'
                            )
                        ),
                        'yaxis': dict(
                            title = 'Veszteség',
                            titlefont = dict(
                                family = 'Helvetica, monospace',
                                size = 12,
                                color = '#7f7f7f'
                            )
                        ),
                        'width':'80%'
                    }
                }
            )),
            html.Center(html.P("A fenti diagrammból az derül ki, hogy e veszteség viszonylag kicsi, ezért elmondható, hogy a modell kis hibával dolgozik.")),
            html.Hr(),
            html.Center(html.P("Itt látható lesz az első modell szerinti pontosság ábrázolása, a Tanítási pontosság, ileltve a Validációs pontosság alapján.")),
            html.Br(),
            html.Center(dcc.Graph(
                id='pontossag-graph',
                figure={
                    'data': [
                        {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'y': [0.3533, 0.4926, 0.5764, 0.6245, 0.6794, 0.7326, 0.7965, 0.8574, 0.9007, 0.9328, 0.9432, 0.9596, 0.9657, 0.9637, 0.9651, 0.9691, 0.9691, 0.9748, 0.9725, 0.9763], 'type': 'scatter', 'name': 'Tanítási pontosság'},
                        {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'y': [0.2756, 0.4737, 0.5492, 0.5573, 0.601, 0.5979, 0.6018, 0.6018, 0.5804, 0.6108, 0.6063, 0.6082, 0.6135, 0.606, 0.6055, 0.6035, 0.6149, 0.6258, 0.6211, 0.6163], 'type': 'scatter', 'name': 'Validációs pontosság'},
                    ],
                    'layout': {
                        'title': 'Pontosság ábrázolása',
                        'xaxis': dict(
                            title = 'Epoch',
                            titlefont = dict(
                                family = 'Helvetica, monospace',
                                size = 12,
                                color = '#7f7f7f'
                            )
                        ),
                        'yaxis': dict(
                            title = 'Pontosság',
                            titlefont = dict(
                                family = 'Helvetica, monospace',
                                size = 12,
                                color = '#7f7f7f'
                            )
                        ),
                        'width':'80%'
                    }
                }
            )),
            html.Center(html.P("Jól látható, hogy a modell nem a legnagyobb pontossággal dolgozik.")),
            ],md=4),
            dbc.Col([html.Center(html.H4("Kayastha Anit munkája")),
            html.Center(html.A("Az osztályozó leírása itt érhető el", href='https://www.kaggle.com/kayasthaanit/aug-fer2013-adadelta-vgg19-e50-cm-roc?fbclid=IwAR15VzqL6Ub-8AUWSmd2HzunUEt6pUYBTMQnUPdMDw4yKCVvaXBO9PK_AxU', style = {'color':'gray', 'font-weight':'bold'})),
            html.Br(),
            html.Center(html.P("Ebben a részben látható lesz a második modell szerinti veszteség ábrázolása, a Tanítási veszteság, ileltve a Validációs veszteség alapján.")),
            html.Br(),
            html.Center(dcc.Graph(
                id='veszteseg2-graph',
                figure={
                    'data': [
                        {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'y': [0.1185, 0.1184, 0.1184, 0.1183, 0.1184, 0.1184, 0.1184, 0.1184, 0.1184, 0.1184, 0.1184, 0.1184, 0.1184, 0.1184, 0.1182, 0.1180, 0.1175, 0.1157, 0.1115, 0.1072], 'type': 'scatter', 'name': 'Tanítási veszteség'},
                        {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'y': [0.1186, 0.1184, 0.1182, 0.1186, 0.1182, 0.1182, 0.1182, 0.1184, 0.1185, 0.1186, 0.1184, 0.1184, 0.1182, 0.1189, 0.1180, 0.1195, 0.1159, 0.1141, 0.1040, 0.1013], 'type': 'scatter', 'name': 'Validációs veszteség'},
                    ],
                    'layout': {
                        'title': 'Veszteség ábrázolása',
                        'xaxis': dict(
                            title = 'Epoch',
                            titlefont = dict(
                                family = 'Helvetica, monospace',
                                size = 12,
                                color = '#7f7f7f'
                            )
                        ),
                        'yaxis': dict(
                            title = 'Veszteség',
                            titlefont = dict(
                                family = 'Helvetica, monospace',
                                size = 12,
                                color = '#7f7f7f'
                            )
                        ),
                        'width':'80%'
                    }
                }
            )),
            html.Center(html.P("A fenti diagrammból az derül ki, hogy e veszteség viszonylag kicsi, ezért elmondható, hogy a modell kis hibával dolgozik.")),
            html.Hr(),
            html.Center(html.P("Itt látható lesz a második modell szerinti pontosság ábrázolása, a Tanítási pontosság, ileltve a Validációs pontosság alapján.")),
            html.Br(),
            html.Center(dcc.Graph(
                id='pontossag2-graph',
                figure={
                    'data': [
                        {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'y': [0.3533, 0.4926, 0.5454, 0.5854, 0.5207, 0.5893, 0.5726, 0.5869, 0.5893, 0.5899, 0.5895, 0.5963, 0.6052, 0.6153, 0.6000, 0.6098, 0.6236, 0.6289, 0.6293, 0.6356], 'type': 'scatter', 'name': 'Tanítási pontosság'},
                        {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'y': [0.5003, 0.3215, 0.5092, 0.4906, 0.5123, 0.5451, 0.5836, 0.5921, 0.5804, 0.5912, 0.5997, 0.6082, 0.6105, 0.6162, 0.6155, 0.6035, 0.6000, 0.5923, 0.5731, 0.6052], 'type': 'scatter', 'name': 'Validációs pontosság'},
                    ],
                    'layout': {
                        'title': 'Pontosság ábrázolása',
                        'xaxis': dict(
                            title = 'Epoch',
                            titlefont = dict(
                                family = 'Helvetica, monospace',
                                size = 12,
                                color = '#7f7f7f'
                            )
                        ),
                        'yaxis': dict(
                            title = 'Pontosság',
                            titlefont = dict(
                                family = 'Helvetica, monospace',
                                size = 12,
                                color = '#7f7f7f'
                            )
                        ),
                        'width':'80%'
                    }
                }
            )),
            html.Center(html.P("Jól látható, hogy a modell az előzőnél is kissebb pontossággal dolgozik.")),
            ],md=4),
            dbc.Col(
                [
                    html.Center(html.H4("Teszt")),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Húzd vagy ',
                        html.A('Válaszd ki a fájlt')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': 'auto'
                    },
                    multiple=True
                ),
                html.Br(),

                html.Div(id='output-image-upload'),
                   
        ]),
        ])
   
], style={'padding': '0px 20px 20px 20px'})


def get_model(path): 
    model = load_model(path)
    print("model loaded")
    return model

global  model1, model2
model1 = get_model("C:/Users/User  A/Desktop/FacialExpressionRecognition/models/model_filter.h5")
model2 = get_model("C:/Users/User  A/Desktop/FacialExpressionRecognition/models/model_keras.h5")

objects = ['Mérges', 'Undorodik', 'Fél', 'Boldog', 'Szomorú', 'Meglepett', 'Semleges']

def predict(model):
    img = image.load_img("C:/Users/User  A/Desktop/FacialExpressionRecognition/Shawon.jpg", grayscale=True, target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    custom = model.predict(x)
    x = np.array(x, 'float32')
    x = x.reshape([48, 48])
    m=0.000000000000000000001
    a=custom[0]
    for i in range(0,len(a)):
        if a[i]>m:
             m=a[i]
             ind=i
    return objects[ind]

def save_picture(contents):
    data = contents.encode("utf8").split(b";base64,")[1]
    with open(os.path.join("./", "asd.png"), "wb") as fp:
        fp.write(base64.decodebytes(data))

def parse_contents(contents, filename, date):
    save_picture(contents)
    pred1 = predict(model1)
    pred2 = predict(model2)
    return html.Div([
        html.Img(src=contents, style={
            'display': 'block',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'width': '42%',
        }),
        html.Br(),
        html.Br(),
        
        html.Center(html.P('Ashadullah Shawon megvalósított osztályozójának eredménye: ')),
        html.Center(html.P(pred1, style={'font-size':'1.5em'})),
        html.Br(),
        html.Center(html.P('Kayastha Anit megvalósított osztályozójának eredménye: ')),
        html.Center(html.P(pred2,style={'font-size':'1.5em'}))
  
       
    
        
    ])




@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



if __name__ == '__main__':
    app.run_server(debug=True)