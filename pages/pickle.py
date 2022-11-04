from dash import Dash, dcc, html, callback, ctx, Input, Output
from dash.exceptions import PreventUpdate

import pandas as pd
import dash


import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
from sklearn import tree
import pandas as pd
import pickle
from explainerdashboard.datasets import titanic_survive
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy, precision, recall, f1
from autosklearn.metrics import (accuracy,
                                 f1,
                                 roc_auc,
                                 precision,
                                 average_precision,
                                 recall,
                                 log_loss, balanced_accuracy, accuracy)
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np
from sklearn.metrics import classification_report
import dash_mantine_components as dmc 

dash.register_page(__name__, title="test")
def error(solution, prediction):
    # custom function defining error
    return np.mean(solution != prediction)


def get_metric_result(cv_results):
    results = pd.DataFrame.from_dict(cv_results)
    results = results[results["status"] == "Success"]
    cols = ["rank_test_scores", "param_classifier:__choice__", "mean_test_score"]
    cols.extend([key for key in cv_results.keys() if key.startswith("metric_")])
    return results[cols]

layout = html.Div([
    dcc.Interval(
                            id='interval_component_analyze_data',
                            interval=10 * 60 * 1000,
                            n_intervals=0),
    html.Hr(),
    html.Center(dmc.Button("Run Model", id='run_model')),
    html.Hr(),
    html.Center(dcc.Loading(html.Div(id='test')))
])

@callback(
    Output('test', 'children'),
    Output('run_model', 'children'),
    Input('run_model', 'n_clicks')
)
def show_result(n):
    print(n)
    if ctx.triggered_id == 'run_model':
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, random_state=1
        )
        error_rate = autosklearn.metrics.make_scorer(
        name="custom_error",
        score_func=error,
        optimum=0,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
        )
        cls = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120,
            per_run_time_limit=30,
            scoring_functions=[balanced_accuracy, precision, recall, f1, error_rate],
        )
        cls.fit(X_train, y_train)
        print(cls.leaderboard())
        
        # from sklearn.datasets import load_iris
        # from sklearn import tree
        # iris = load_iris()
        # X, y = iris.data, iris.target
        # clf = tree.DecisionTreeClassifier()
        # clf = clf.fit(X_train, y_train)
        # print(clf.predict(X_test))
        with open('experiment_name.pkl', 'wb') as f:
                pickle.dump(cls, f)

        return html.Div('Done'), dash.no_update
    else:
        raise PreventUpdate

