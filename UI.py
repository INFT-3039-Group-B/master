import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import base64
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from scipy.interpolate import make_interp_spline
import numpy as np

# Create a Dash web application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])



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
            dcc.Graph(id='visualization-graph'),
        ], className="tab-content", style={'background-color': '#EFFBFB', 'padding': '20px', 'border-radius': '10px'}),
    ]),
])

@app.callback(
    Output("download-data", "data"),
    Input("export-processed-button", "n_clicks"),
    Input("upload-data", "filename"),
    Input("upload-data", "contents"),
    prevent_initial_call=True,
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
    Output('output-sentiment', 'children'),
    Input('analyze-button', 'n_clicks'),
    State('input-text', 'value')
)
def analyze_sentiment(n_clicks, input_text):
    # Perform sentiment analysis here and return the result
    return "Sentiment: Positive"  # Replace with your analysis result

def download_processed_data(n_clicks):
    # Replace this with the processed data download logic
    return {"content": b"", "filename": "processed_data.csv"}

@app.callback(
    Output('visualization-graph', 'figure'),
    Input('visualization-dropdown', 'value')
)
def update_visualization(selected_option):
    if selected_option == 'word-cloud':
        # Generate and return a Word Cloud visualization
        wordcloud_data = generate_wordcloud()
        return wordcloud_data
    elif selected_option == 'network-graph':
        # Generate and return a Network Graph visualization
        network_data = generate_network_graph()
        return network_data
    elif selected_option == 'time-series':
        # Generate and return a Time Series visualization
        time_series_data = generate_time_series()
        return time_series_data
    elif selected_option == 'tree-map':
        # Generate and return a Tree Map visualization
        tree_map_data = generate_tree_map()
        return tree_map_data
    elif selected_option == 'pie-chart':
        # Generate and return a Pie Chart visualization
        pie_chart_data = generate_pie_chart()
        return pie_chart_data
    
def generate_wordcloud():
    # Generate Word Cloud data here
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


def generate_time_series():
    df = pd.read_csv('master_date_scoreSumed.csv')  # Replace with dataset file
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')

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


def generate_tree_map():

    df = pd.read_csv('1191 preprocessed.csv')  # Replace with dataset file
    # Create a Tree Map
    df_2000 = df[df['Year'] == 2000]

    # Create a Tree Map for the year 2000
    tree_map_fig = px.treemap(df_2000, path=['Year', 'Label'], color='Label')
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


if __name__ == '__main__':
    app.run_server(debug=True,port=8050)
