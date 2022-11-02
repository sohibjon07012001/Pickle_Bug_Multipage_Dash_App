import dash
from dash import Input, Output, dcc, html, dash_table
import pandas as pd
from pycaret.classification import *
import dash_bootstrap_components as dbc


application = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True, prevent_initial_callbacks=True, external_stylesheets=[dbc.themes.MATERIA, dbc.icons.FONT_AWESOME])
app = application.server


sidebar = html.Div(
    [
        html.Div(
            [
                # width: 3rem ensures the logo is the exact width of the
                # collapsed sidebar (accounting for padding)
                html.H2("Pickle", style={'color': 'black'}),
            ],
            className="sidebar-header",
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-home me-2"),
                     html.Span("Pickle")],
                    href=dash.page_registry['pages.pickle']['path'],
                    active="exact",
                ),
            
                
                
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar"
)


application.layout = html.Div([
                      sidebar,
                      dash.page_container,

])


if __name__ == "__main__":
    application.run_server(debug=True)
