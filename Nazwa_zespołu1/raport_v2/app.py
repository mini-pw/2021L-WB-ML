import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from sklearn.model_selection import train_test_split
import xgboost as xgb
import plotly.figure_factory as ff
from functions import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

dt = pd.read_csv('all_data.csv')
dt = dt[['(%)lymphocyte', 'High sensitivity C-reactive protein', 'Lactate dehydrogenase', 'outcome']]

fig = ff.create_distplot([dt['(%)lymphocyte'].dropna().to_numpy()], ['lymphocytes'])
fig2 = ff.create_distplot([dt['High sensitivity C-reactive protein'].dropna().to_numpy()], ['hs-CRP'])
fig3 = ff.create_distplot([dt['Lactate dehydrogenase'].dropna().to_numpy()], ['LDH'])

app.layout = html.Div(children=[
    html.H1(children='Interactive survival prediction model'),
    html.Div(children=[
        html.Label("Choose your patient's nationality:"),
        dcc.RadioItems(
            id='nation',
            options=[{'label': i, 'value': i} for i in ['Chinese', 'Dutch', 'American', 'French', 'Else']],
            value='Else',
            style={'display': 'block'})]),

    html.Div(id='my-output2'),
    html.Div(id='my-output3'),
    html.Div(
        id='slider1', children=
        ["Give value of your patient's lymphocytes",
         dcc.Slider(
             id='val1',
             min=0,
             max=95,
             step=1,
             value=40,
             marks={i: '{}'.format(i) for i in range(0, 96, 10)}
         )],
        style={'display': 'block'}),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    html.Div(
        id='slider2', children=
        ["Give value of your patient's hs-CRP",
         dcc.Slider(
             id='val2',
             min=0,
             max=600,
             step=5,
             value=300,
             marks={i: '{}'.format(i) for i in range(0, 600, 50)}
         )],
        style={'display': 'block'}),
    dcc.Graph(
        id='example-graph2',
        figure=fig2
    ),
    html.Div(
        id='slider3', children=
        ["Give value of your patient's LDH",
         dcc.Slider(
             id='val3',
             min=90,
             max=5000,
             step=20,
             value=2000,
             marks={i: '{}'.format(i) for i in range(0, 5000, 250)}
         )],
        style={'display': 'block'}),

    dcc.Graph(
        id='example-graph3',
        figure=fig3
    ),
    html.Br(),
    html.Div(id='my-output'),
    html.Div(id='my-output-df2'),
    html.Div(id='my-output-df', style={'font-size': '50px'})

], style={'margin-top': '10px',
          'margin-bottom': '5px',
          'text-align': 'left',
          'paddingLeft': 5,
          'background': 'lightgreen'})


@app.callback(
    [Output(component_id='my-output', component_property='children'),
     Output(component_id='my-output-df2', component_property='children'),
     Output(component_id='my-output3', component_property='children'),
     Output(component_id='my-output-df', component_property='children'),
     Output(component_id='example-graph', component_property='figure'),
     Output(component_id='example-graph2', component_property='figure'),
     Output(component_id='example-graph3', component_property='figure')
     ],
    [Input(component_id='nation', component_property='value'),
     Input(component_id='val1', component_property='value'),
     Input(component_id='val2', component_property='value'),
     Input(component_id='val3', component_property='value')]
)
def update_output_div(input_value, val1, val2, val3):
    if input_value == 'American':
        dt = pd.read_csv('american_data.csv')
    elif input_value == 'Dutch':
        dt = pd.read_csv('dutch_data.csv')
    elif input_value == 'Chinese':
        dt = pd.read_csv('chinese_data.csv')
    elif input_value == 'French':
        dt = pd.read_csv('french_data.csv')
    else:
        dt = pd.read_csv('all_data.csv')
    dt = dt[['(%)lymphocyte', 'High sensitivity C-reactive protein', 'Lactate dehydrogenase', 'outcome']]
    dt['outcome'] = dt['outcome'].astype('int')
    x_train, x_test, y_train, y_test = train_test_split(dt.drop('outcome', axis=1), dt.outcome, test_size=0.3,
                                                        random_state=1)
    model = xgb.XGBClassifier(
        max_depth=4
        , learning_rate=0.2
        , reg_lambda=1
        , n_estimators=150
        , subsample=0.9
        , colsample_bytree=0.9
        , use_label_encoder=False)
    model.fit(x_train, y_train)
    return_value = 'The patient will probably survive!'
    if int(model.predict(pd.DataFrame(data={'(%)lymphocyte': [val1],
                                                                         'High sensitivity C-reactive protein': [val2],
                                                                         'Lactate dehydrogenase': [val3]}))) == 1:
        return_value = 'The patient will probably die!'
    return 'Probability of survival: {}'.format(model.predict_proba(pd.DataFrame(data={'(%)lymphocyte': [val1],
                                                                                       'High sensitivity C-reactive protein': [
                                                                                           val2],
                                                                                       'Lactate dehydrogenase': [
                                                                                           val3]}))[0][0]
                                                ), \
           'Probability of death: {}'.format(model.predict_proba(pd.DataFrame(data={'(%)lymphocyte': [val1],
                                                                                    'High sensitivity C-reactive protein': [
                                                                                        val2],
                                                                                    'Lactate dehydrogenase': [val3]}))[0][1]
                                             ), \
           "Accuracy on test set of the model: {}".format(float(np.mean(model.predict(x_test) == y_test))),\
           return_value, \
           ff.create_distplot([dt['(%)lymphocyte'].dropna().to_numpy()], ['lymphocytes']), \
           ff.create_distplot([dt['High sensitivity C-reactive protein'].dropna().to_numpy()], ['hs-CRP']), \
           ff.create_distplot([dt['Lactate dehydrogenase'].dropna().to_numpy()], ['LDH'])


@app.callback([Output(component_id='val1', component_property='min'),
               Output(component_id='val1', component_property='max'),
               Output(component_id='val1', component_property='value'),
               Output(component_id='val2', component_property='min'),
               Output(component_id='val2', component_property='max'),
               Output(component_id='val2', component_property='value'),
               Output(component_id='val3', component_property='min'),
               Output(component_id='val3', component_property='max'),
               Output(component_id='val3', component_property='value')
               ],
              Input(component_id='nation', component_property='value'))
def update_slider(input_value):
    if input_value == 'American':
        dt = pd.read_csv('american_data.csv')
    elif input_value == 'Dutch':
        dt = pd.read_csv('dutch_data.csv')
    elif input_value == 'Chinese':
        dt = pd.read_csv('chinese_data.csv')
    elif input_value == 'French':
        dt = pd.read_csv('french_data.csv')
    else:
        dt = pd.read_csv('all_data.csv')
    return int(np.min(dt['(%)lymphocyte'].dropna())), int(np.max(dt['(%)lymphocyte'].dropna())) + 5,\
           round(int(np.max(dt['(%)lymphocyte'].dropna()))/2, 10),\
           int(np.min(dt['High sensitivity C-reactive protein'].dropna())), int(np.max(dt['High sensitivity C-reactive protein'].dropna())) + 5, \
           round(int(np.max(dt['High sensitivity C-reactive protein'].dropna()))/2, 10), \
           int(np.min(dt['Lactate dehydrogenase'].dropna())), int(np.max(dt['Lactate dehydrogenase'].dropna())) + 5,\
           round(int(np.max(dt['Lactate dehydrogenase'].dropna()))/2, 10),


if __name__ == '__main__':
    app.run_server(debug=False)
