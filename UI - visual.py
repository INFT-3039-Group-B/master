import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import base64
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.interpolate import make_interp_spline
import numpy as np
import base64
import io

# Create a Dash web application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load your dataset
df = pd.read_csv('1191 preprocessed.csv')  # Replace with your dataset file

df_wordCloud = pd.read_csv('1191_cleaned.csv')  # Replace with your dataset file

df_time_serie = pd.read_csv('master_date_scoreSumed.csv')  # Replace with dataset file
df_time_serie['Date'] = pd.to_datetime(df_time_serie['Date'], format='%Y-%m-%d', errors='coerce')

# Get unique years from your dataset
available_years = ['All Years'] + df['Year'].unique().tolist()

# Define CSS styles to enhance the appearance
app.layout = dbc.Tabs([
    dbc.Tab(label="Load and Preprocess", children=[
        html.Div([
            html.H1("Load and Preprocess", style={'color': '#0080FF', 'font-size': '36px'}),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                }
            ),
            dcc.Download(id="download-data"),
            # Add more content for this scenario
        ], className="tab-content", style={'background-color': '#EFFBFB', 'padding': '20px', 'border-radius': '10px'}),
    ]),
    dbc.Tab(label="Import Pre-Processed Emails", children=[
        html.Div([
            html.H1("Import Pre-Processed Emails", style={'color': '#0080FF', 'font-size': '36px'}),
            dcc.Upload(
                id='upload-processed-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                }
            ),
            dcc.Download(id="download-processed-data"),
            # Add more content for this scenario
        ], className="tab-content", style={'background-color': '#EFFBFB', 'padding': '20px', 'border-radius': '10px'}),
    ]),
    dbc.Tab(label="Sentiment Classification", children=[
        html.Div([
            html.H1("Sentiment Classification", style={'color': '#0080FF', 'font-size': '36px'}),
            dcc.Upload(
                id='upload-sentiment-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                }
            ),
            dcc.Download(id="download-sentiment-data"),
            dcc.Textarea(id='input-text', placeholder='Enter text for sentiment analysis...'),
            html.Div([
                html.Button('Analyze Sentiment', id='analyze-button', style={'margin-top': '10px'}),
                html.Div(id='output-sentiment'),
            ]),
            # Add more content for this scenario
        ], className="tab-content", style={'background-color': '#EFFBFB', 'padding': '20px', 'border-radius': '10px'}),
    ]),
    dbc.Tab(label="Export Emails", children=[
        html.Div([
            html.H1("Export Emails", style={'color': '#0080FF', 'font-size': '36px'}),
            html.Button("Export Processed Data", id="export-processed-button", n_clicks=0),
            # Add more content for this scenario
        ], className="tab-content", style={'background-color': '#EFFBFB', 'padding': '20px', 'border-radius': '10px'}),
    ]),
    dbc.Tab(label="Visualization", children=[
        html.Div([
            html.H1("Visualization", style={'color': '#0080FF', 'font-size': '36px'}),
            dcc.Dropdown(
                id='visualization-dropdown',
                options=[
                    {'label': 'Word Cloud', 'value': 'word-cloud'},
                    {'label': 'Network Graph', 'value': 'network-graph'},
                    {'label': 'Time Series', 'value': 'time-series'},
                    {'label': 'Tree Map', 'value': 'tree-map'},
                    {'label': 'Pie Chart', 'value': 'pie-chart'}
                ],
                value='word-cloud'
            ),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': year, 'value': year} for year in available_years],
                    value='All Years'  # Default to "All Years"
                )),
            ]),
            dcc.Graph(
                id='visualization-graph',
                style={'width': '100%', 'height': '700px'},  # Adjust the height as needed
            ),
        ], style={'background-color': '#EFFBFB', 'padding': '20px', 'border-radius': '10px'}),
    ]),
])

@app.callback(
    Output("download-data", "data"),
    Input("upload-data", "filename"),
    Input("upload-data", "contents"),
)
def download_uploaded_data(filename, contents):
    if filename is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return {"content": decoded, "filename": filename}

@app.callback(
    Output("download-processed-data", "data"),
    Input("upload-processed-data", "filename"),
    Input("upload-processed-data", "contents"),
)
def download_uploaded_processed_data(filename, contents):
    if filename is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return {"content": decoded, "filename": filename}

@app.callback(
    Output("download-sentiment-data", "data"),
    Input("upload-sentiment-data", "filename"),
    Input("upload-sentiment-data", "contents"),
)
def download_uploaded_sentiment_data(filename, contents):
    if filename is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return {"content": decoded, "filename": filename}

@app.callback(
    Output('visualization-graph', 'figure'),
    Input('visualization-dropdown', 'value'),
    Input('year-dropdown', 'value')
)
def update_visualization(selected_option, selected_year):

    if selected_year == 'All Years':
        filtered_df = df  # No filtering by year
        filtered_df_time_serie = df_time_serie  # # No filtering by year
    else:
        filtered_df = df[df['Year'] == selected_year]
        filtered_df_time_serie = df_time_serie[df_time_serie['Date'].dt.year == selected_year]

    if selected_option == 'word-cloud':

        df_wordCloud['Text'] = df_wordCloud['Text'].apply(str_text)

        all_text = ' '.join(df_wordCloud['Text'])

        wordcloud_data = generate_wordcloud(all_text)

        return wordcloud_data

    elif selected_option == 'network-graph':
        # Generate and return a Network Graph visualization
        network_data = generate_network_graph()
        return network_data
    elif selected_option == 'time-series':
        # Generate and return a Time Series visualization
        time_series_data = generate_time_series(filtered_df_time_serie)
        return time_series_data
    elif selected_option == 'tree-map':
        # Generate and return a Tree Map visualization
        tree_map_data = generate_tree_map(filtered_df)
        return tree_map_data
    elif selected_option == 'pie-chart':
        # Generate and return a Pie Chart visualization
        pie_chart_data = generate_pie_chart()
        return pie_chart_data

def str_text(text):
    text = str(text)
    return text



def generate_wordcloud(content):

    # Generate the word cloud
    wordcloud = WordCloud(width=1000, height=600, background_color="white").generate(content)

    # Create a BytesIO buffer to save the word cloud image
    buffer = io.BytesIO()
    
    # Save the word cloud image to the buffer
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    
    # Encode the image as base64
    wordcloud_base64 = base64.b64encode(buffer.read()).decode()

    return {
        'data': [{
            'x': [0],
            'y': [0],
            'mode': 'text',
            'text': ['Word Cloud'],
            'textfont': {
                'size': 24,
                'color': 'black'  # Customize text color
            }
        }],
        'layout': {
            'images': [
                {
                    'source': 'data:image/png;base64,{}'.format(wordcloud_base64),
                    'x': 0,
                    'y': 0,
                    'xref': 'x',
                    'yref': 'y',
                    'sizex': 1,
                    'sizey': 1,
                    'xanchor': 'center',
                    'yanchor': 'middle'
                }
            ],
            'width': 1000,
            'height': 600,
            'xaxis': {
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False
            },
            'yaxis': {
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False
            }
        }
    }

def generate_network_graph():
    # Generate Network Graph data here
        # Generate and return a Pie Chart visualization
    df = pd.read_csv('master_date_score.csv')  # Replace with your dataset file
    label_counts = df['label'].value_counts()

    labels = label_counts.index
    values = label_counts.values

    pie_chart_data = {
        'data': [{
            'type': 'pie',
            'labels': labels,
            'values': values,
        }],
        'layout': {
            'title': 'Distribution of Labels'
        }
    }
    return pie_chart_data


def generate_time_series(df):


    # Sort the DataFrame by 'Date' to ensure it's in the right order for plotting
    df = df.sort_values(by='Date')
    

    # Create a new DataFrame for the smoothed curve
    smooth_df = pd.DataFrame()
    num_dates = np.linspace(0, 1, len(df))  # Create a linear space of numerical dates between 0 and 1

    # Create a spline function
    spline = make_interp_spline(num_dates, df['score'], k=3)

    # Generate the numerical dates for the smoothed curve
    num_dates_smooth = np.linspace(0, 1, 100)  # Adjust the number of points (100 in this example)
    
    # Get the smoothed scores for the numerical dates
    scores_smooth = spline(num_dates_smooth)

    # Convert numerical dates back to actual dates
    smooth_dates = df['Date'].min() + pd.to_timedelta(num_dates_smooth * (df['Date'].max() - df['Date'].min()))

    smooth_df['Date'] = smooth_dates
    smooth_df['score'] = scores_smooth

    fig = px.scatter(
        df, x='Date', y='score',
        title='Score by Date'
    )

    # Add the smoothed curve
    fig.add_scatter(x=smooth_df['Date'], y=smooth_df['score'], mode='lines', name='Smoothed Curve')

    return fig


def generate_tree_map(df):

    # Create a Tree Map for the selected year
    tree_map_fig = px.treemap(df, path=['Year', 'Label'], color='Label')
    return tree_map_fig

def generate_pie_chart():
    # Generate and return a Pie Chart visualization
    df = pd.read_csv('master_date_score.csv')  # Replace with dataset file
    label_counts = df['label'].value_counts()

    labels = label_counts.index
    values = label_counts.values

    pie_chart_data = {
        'data': [{
            'type': 'pie',
            'labels': labels,
            'values': values,
        }],
        'layout': {
            'title': 'Distribution of Labels'
        }
    }
    return pie_chart_data    


@app.callback(
    Output('output-sentiment', 'children'),
    Input('analyze-button', 'n_clicks'),
    State('input-text', 'value')
)
def analyze_sentiment(n_clicks, input_text):
    # Perform sentiment analysis here and return the result
    return "Sentiment: Positive"  # Replace with your analysis result

if __name__ == '__main__':
    app.run_server(debug = True, threaded=True, port=8052)  # By setting threaded=True, 
                                                            # indicating that the Dash app should be run in a multi-threaded mode, 
                                                            # and it may help mitigate the warning related to Matplotlib. 
