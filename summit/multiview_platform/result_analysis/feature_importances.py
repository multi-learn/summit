import os

import numpy as np
import pandas as pd
import plotly

from ..monoview.monoview_utils import MonoviewResult


def get_feature_importances(result, feature_ids=None, view_names=None,):
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
                    index=feature_ids[classifier_result.view_index])
            if hasattr(classifier_result.clf, 'feature_importances_'):
                feature_importances[classifier_result.view_name][
                    classifier_result.classifier_name] = classifier_result.clf.feature_importances_
            else:
                feature_importances[classifier_result.view_name][
                    classifier_result.classifier_name] = np.zeros(
                    classifier_result.n_features)
        else:
            if "mv" not in feature_importances:
                feat_ids = []
                for view_ind, v_feature_id in enumerate(feature_ids):
                    feat_ids += [view_names[view_ind] + "-" + ind for ind in
                                 v_feature_id]
                feature_importances["mv"] = pd.DataFrame(index=feat_ids)
            if hasattr(classifier_result.clf, 'feature_importances_'):
                feature_importances["mv"][classifier_result.classifier_name] = classifier_result.clf.feature_importances_
    return feature_importances


def publish_feature_importances(feature_importances, directory, database_name,
                                feature_stds=None, metric_scores=None):  # pragma: no cover
    importance_dfs = []
    std_dfs = []
    if not os.path.exists(os.path.join(directory, "feature_importances")):
        os.mkdir(os.path.join(directory, "feature_importances"))
    for view_name, feature_importance in feature_importances.items():
        if view_name!="mv":

            if feature_stds is not None:
                feature_std = feature_stds[view_name]
            else:
                feature_std = pd.DataFrame(data=np.zeros(feature_importance.shape),
                                           index=feature_importance.index,
                                           columns=feature_importance.columns)
            feature_std = feature_std.loc[feature_importance.index]


            importance_dfs.append(feature_importance.set_index(pd.Index([view_name+"-"+ind for ind in list(feature_importance.index)])))
            # importance_dfs.append(pd.DataFrame(index=[view_name+"-br"],
            #                                    columns=feature_importance.columns,
            #                                    data=np.zeros((1, len(
            #                                        feature_importance.columns)))))
            std_dfs.append(feature_std.set_index(pd.Index([view_name+"-"+ind
                                                           for ind
                                                           in list(feature_std.index)])))
            # std_dfs.append(pd.DataFrame(index=[view_name + "-br"],
            #                                    columns=feature_std.columns,
            #                                    data=np.zeros((1, len(
            #                                        feature_std.columns)))))
    if len(importance_dfs)>0:
        feature_importances_df = pd.concat(importance_dfs)
        feature_importances_df = feature_importances_df/feature_importances_df.sum(axis=0)

        feature_std_df = pd.concat(std_dfs)
        if "mv" in feature_importances:
            feature_importances_df = pd.concat([feature_importances_df,feature_importances["mv"].loc[(feature_importances["mv"] != 0).any(axis=1), :]], axis=1).fillna(0)
            if feature_stds is not None:
                feature_std_df = pd.concat([feature_std_df, feature_stds["mv"]], axis=1,).fillna(0)
            else:
                fake = pd.DataFrame(data=np.zeros((feature_importances_df.shape[0], feature_importances["mv"].shape[1])),
                                               index=feature_importances_df.index,
                                               columns=feature_importances["mv"].columns).fillna(0)
                feature_std_df = pd.concat([feature_std_df, fake], axis=1,).fillna(0)
        plot_feature_importances(os.path.join(directory, "feature_importances",
                                     database_name), feature_importances_df, feature_std_df)
        if metric_scores is not None:
            plot_feature_relevance(os.path.join(directory, "feature_importances",
                                     database_name), feature_importances_df, feature_std_df, metric_scores)


def plot_feature_importances(file_name, feature_importance,
                             feature_std):  # pragma: no cover
    s = feature_importance.sum(axis=1)
    s = s[s!=0]
    feature_importance = feature_importance.loc[s.sort_values(ascending=False).index]
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
        colorscale="Hot",
        reversescale=True))
    fig.update_layout(
        xaxis={"showgrid": False, "showticklabels": False, "ticks": ''},
        yaxis={"showgrid": False, "showticklabels": False, "ticks": ''})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    plotly.offline.plot(fig, filename=file_name + ".html", auto_open=False)

    del fig
    

def plot_feature_relevance(file_name, feature_importance,
                             feature_std, metric_scores): # pragma: no cover
    for metric, score_df in metric_scores.items():
        if metric.endswith("*"):
            if isinstance(score_df, dict):
                score_df = score_df["mean"]
            for score in score_df.columns:
                if len(score.split("-"))>1:
                    algo, view = score.split("-")
                    feature_importance[algo].loc[[ind for ind in feature_importance.index if ind.startswith(view)]]*=score_df[score]['test']
                else:
                    feature_importance[score] *= score_df[score]['test']
    file_name+="_relevance"
    plot_feature_importances(file_name, feature_importance,
                             feature_std)
