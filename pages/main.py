from subprocess import call
import dash_bootstrap_components as dbc
import dash
from dash import html, dcc, dash_table, callback, Input, Output, State, long_callback, ALL, MATCH, ctx




dash.register_page(__name__, path_template="/")



layout = html.Div([
    html.Center(dcc.Link(href='/pickle', children="Pickle page")),
    
])