import os

import pandas as pd
import plotly


def get_duration(results):
    df = pd.DataFrame(columns=["hps", "fit", "pred"], )
    for classifier_result in results:
        df.at[classifier_result.get_classifier_name(),
              "hps"] = classifier_result.hps_duration
        df.at[classifier_result.get_classifier_name(),
              "fit"] = classifier_result.fit_duration
        df.at[classifier_result.get_classifier_name(),
              "pred"] = classifier_result.pred_duration
    return df


def plot_durations(durations, directory, database_name,
                   durations_stds=None):  # pragma: no cover
    file_name = os.path.join(directory, database_name + "-durations")
    durations.to_csv(file_name + "_dataframe.csv")
    fig = plotly.graph_objs.Figure()
    if durations_stds is None:
        durations_stds = pd.DataFrame(0, durations.index, durations.columns)
    else:
        durations_stds.to_csv(file_name + "_stds_dataframe.csv")
    fig.add_trace(plotly.graph_objs.Bar(name='Hyper-parameter Optimization',
                                        x=durations.index,
                                        y=durations['hps'],
                                        error_y=dict(type='data',
                                                     array=durations_stds[
                                                         "hps"]),
                                        marker_color="grey"))
    fig.add_trace(plotly.graph_objs.Bar(name='Fit (on train set)',
                                        x=durations.index,
                                        y=durations['fit'],
                                        error_y=dict(type='data',
                                                     array=durations_stds[
                                                         "fit"]),
                                        marker_color="black"))
    fig.add_trace(plotly.graph_objs.Bar(name='Prediction (on test set)',
                                        x=durations.index,
                                        y=durations['pred'],
                                        error_y=dict(type='data',
                                                     array=durations_stds[
                                                         "pred"]),
                                        marker_color="lightgrey"))
    fig.update_layout(title="Durations for each classfier",
                      yaxis_title="Duration (s)")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    plotly.offline.plot(fig, filename=file_name + ".html", auto_open=False)
