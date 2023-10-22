import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load your CSV data
df = pd.read_csv('master_date_scoreSumed.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')

# Filter out NaN values in the 'Date' column
#df = df.dropna(subset=['Date'])

# Create a Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year), 'value': year} for year in df['Date'].dt.year.unique()],
        value=df['Date'].dt.year.max(),
        multi=False
    ),
    dcc.Graph(id='score-plot')
])

# Define callback to update the plot based on selected year
@app.callback(
    Output('score-plot', 'figure'),
    Input('year-dropdown', 'value')
)
def update_plot(selected_year):
    filtered_df = df[df['Date'].dt.year == selected_year]
    fig = px.scatter(
        filtered_df, x='Date', y='score',
        title=f'Score by Date for {selected_year}',
        trendline="ols"  # Add the trend line
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
