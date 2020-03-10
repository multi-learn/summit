import os
import plotly
import pandas as pd
import numpy as np

from ..monoview.monoview_utils import MonoviewResult


def get_feature_importances(result, feature_names=None):
    r"""Extracts the feature importance from the monoview results and stores
    them in a dictionnary :
    feature_importance[view_name] is a pandas.DataFrame of size n_feature*n_clf
    containing a score of importance for each feature.

    Parameters
    ----------
    result : list of results

    Returns
    -------
    feature_importances : dict of pd.DataFrame
        The dictionary containing all the feature importance for each view as
        pandas DataFrames
    """
    feature_importances = {}
    for classifier_result in result:
        if isinstance(classifier_result, MonoviewResult):
            if classifier_result.view_name not in feature_importances:
                feature_importances[classifier_result.view_name] = pd.DataFrame(
                    index=feature_names)
            if hasattr(classifier_result.clf, 'feature_importances_'):
                feature_importances[classifier_result.view_name][
                    classifier_result.classifier_name] = classifier_result.clf.feature_importances_
            else:
                feature_importances[classifier_result.view_name][
                    classifier_result.classifier_name] = np.zeros(
                    classifier_result.n_features)
    return feature_importances

def publish_feature_importances(feature_importances, directory, database_name,
                                feature_stds=None):
    for view_name, feature_importance in feature_importances.items():
        if not os.path.exists(os.path.join(directory, "feature_importances")):
            os.mkdir(os.path.join(directory, "feature_importances"))
        file_name = os.path.join(directory, "feature_importances",
                                 database_name + "-" + view_name
                                 + "-feature_importances")
        if feature_stds is not None:
            feature_std = feature_stds[view_name]
            feature_std.to_csv(file_name + "_dataframe_stds.csv")
        else:
            feature_std = pd.DataFrame(data=np.zeros(feature_importance.shape),
                                       index=feature_importance.index,
                                       columns=feature_importance.columns)
        feature_importance.to_csv(file_name + "_dataframe.csv")
        hover_text = [["-Feature :" + str(feature_name) +
                       "<br>-Classifier : " + classifier_name +
                       "<br>-Importance : " + str(
            feature_importance.loc[feature_name][classifier_name]) +
                       "<br>-STD : " + str(
            feature_std.loc[feature_name][classifier_name])
                       for classifier_name in list(feature_importance.columns)]
                      for feature_name in list(feature_importance.index)]
        fig = plotly.graph_objs.Figure(data=plotly.graph_objs.Heatmap(
            x=list(feature_importance.columns),
            y=list(feature_importance.index),
            z=feature_importance.values,
            text=hover_text,
            hoverinfo=["text"],
            colorscale="Greys",
            reversescale=False))
        fig.update_layout(
            xaxis={"showgrid": False, "showticklabels": False, "ticks": ''},
            yaxis={"showgrid": False, "showticklabels": False, "ticks": ''})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        plotly.offline.plot(fig, filename=file_name + ".html", auto_open=False)

        del fig

