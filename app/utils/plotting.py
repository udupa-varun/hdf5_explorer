import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------
# Generic/Common
# ----------------

# max number of markers per chart
# if exceeded, figure will force line charts
MAX_MARKERS_PER_FIGURE = int(1e4)

# default colors for health thresholds
DEFAULT_WARNING_COLOR = "#FFA500"
DEFAULT_ALARM_COLOR = "#FF0000"

# max number of allowable rows for the features table
FEAT_TABLE_ROW_LIMIT = int(1e3)
FEAT_TABLE_HIGHLIGHT_COLOR = "#ff4b4b"

chart_types = {
    "Line": "lines",
    "Scatter-Line": "lines+markers",
    "Scatter": "markers",
}

chart_config = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["zoomIn", "zoomOut"],
    "displaylogo": False,
    "toImageButtonOptions": {
        "height": None,
        "width": None,
    },
}


def update_figure(fig: go.Figure) -> go.Figure:
    """Layout updates to be applied to ALL figures generated.

    :param fig: plotly figure object to be updated.
    :type fig: go.Figure
    :return: updated plotly figure object.
    :rtype: go.Figure
    """
    # hover label precision
    fig.update_traces(hovertemplate="%{y:.4g}")
    # title and legend updates
    fig.update_layout(
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="right",
            y=1.02,
            x=1,
        ),
        hovermode="x unified",
    )
    # updates for X axis
    fig.update_layout(
        xaxis_showline=True,
        xaxis_mirror="ticks",
        xaxis_linewidth=1,
        xaxis_ticks="inside",
        xaxis_ticklen=5,
        xaxis_tickwidth=1,
    )
    # updates for Y axis
    fig.update_layout(
        yaxis_showline=True,
        yaxis_mirror="ticks",
        yaxis_linewidth=1,
        yaxis_ticks="inside",
        yaxis_ticklen=5,
        yaxis_tickwidth=1,
    )
    # updates for marker-based performance
    fig = handle_marker_safety(fig)

    return fig


def handle_marker_safety(fig: go.Figure) -> go.Figure:
    """Checks the supplied figure to ensure plotting
    large number of data points is handled safely.

    :param fig: input figure
    :type fig: go.Figure
    :return: modified figure that is (hopefully) more performant
    :rtype: go.Figure
    """
    num_data_points = get_data_point_count(fig)

    # get chart type requested
    chart_mode = fig["data"][0]["mode"]
    # check if we are "marker-safe"
    if num_data_points > MAX_MARKERS_PER_FIGURE:
        # we are not marker-safe,
        # so override chart type with a translucent line plot
        # TODO: remove this warning? can get annoying for multiple figures
        st.warning(
            "Too many data points! Forcing a line chart and disabling hover labels."
        )
        fig.update_traces(
            mode="lines",
            opacity=0.3,
        )
        # also disable hover labels
        fig.update_layout(hovermode=False)
    # safe to add marker borders
    elif "markers" in chart_mode:
        fig.update_traces(
            marker=dict(
                size=10,
                line=dict(width=1, color="DarkSlateGrey"),
            )
        )

    return fig


def get_data_point_count(fig: go.Figure) -> int:
    """Get total number of data points in supplied figure.
    Note that this includes all traces for the current figure.
    Assumes a common X axis for all traces.

    :param fig: figure to be inspected
    :type fig: go.Figure
    :return: number of data points present in figure data
    :rtype: int
    """
    num_data_points: int = 0
    # get from X data when possible as shorter X can truncate Y
    # sometimes X data is not present (Index) so use Y data instead
    axis = "x" if fig["data"][0]["x"] is not None else "y"
    num_data_points += np.sum(len(d[axis]) for d in fig["data"])

    return num_data_points


# ----------------
# Health Tab
# ----------------


def plot_health(health_df: pd.DataFrame, datetime_chunk: np.ndarray):
    """Plots one or more charts for the health tab.

    :param health_df: dataframe with health values.
    :type health_df: pd.DataFrame
    :param datetime_chunk: array of record timestamps
    :type datetime_chunk: np.ndarray
    """
    separate_charts = st.session_state["separate_health_charts"]

    if separate_charts:
        plot_health_separate(health_df, datetime_chunk)
    else:
        plot_health_single(health_df, datetime_chunk)


def plot_health_single(health_df: pd.DataFrame, datetime_chunk: np.ndarray):
    """Plots a single health chart with all the selected health components.

    :param health_df: dataframe with health values.
    :type health_df: pd.DataFrame
    :param datetime_chunk: array of record timestamps
    :type datetime_chunk: np.ndarray
    """
    # init figure
    fig = go.Figure()
    max_health_val: float = 0.0
    for health_component in health_df:
        # compute trace
        trace = go.Scattergl(
            x=datetime_chunk,
            y=health_df[health_component],
            name=health_component,
            mode="lines",
        )
        # add trace to figure
        fig.add_trace(trace)
        # store max value
        max_health_val = max(max_health_val, max(health_df[health_component]))
    # common updates to figure
    fig = update_figure(fig)

    fig = plot_threshold_lines(fig, max_health_val=max_health_val)
    # specific updates to figure
    fig.update_layout(
        title_x=0.5,
        title_text="Health Index",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        showlegend=True,
    )
    # render figure in app
    st.plotly_chart(fig, use_container_width=True, config=chart_config)


def plot_health_separate(health_df: pd.DataFrame, datetime_chunk: np.ndarray):
    """Plots multiple health charts, one for each selected health component.

    :param health_df: dataframe with health values.
    :type health_df: pd.DataFrame
    :param datetime_chunk: array of record timestamps
    :type datetime_chunk: np.ndarray
    """
    for health_component in health_df:
        # init figure
        fig = go.Figure()
        # compute trace
        trace = go.Scattergl(
            x=datetime_chunk,
            y=health_df[health_component],
            name=health_component,
            mode="lines",
        )
        # add trace to figure
        fig.add_trace(trace)
        # common updates to figure
        fig = update_figure(fig)
        # specific updates to figure
        fig = plot_threshold_lines(fig, max_health_val=max(health_df[health_component]))
        fig.update_layout(
            title_x=0.5,
            title_text=health_component,
            xaxis_title="Timestamp",
            yaxis_title="Health Index",
            showlegend=False,
        )
        # render figure in app
        st.plotly_chart(fig, use_container_width=True, config=chart_config)


def plot_threshold_lines(fig: go.Figure, max_health_val: float) -> go.Figure:
    """Plots horizontal lines on the given figure object,
    based on the configured threshold controls.

    :param fig: plotly figure object.
    :type fig: go.Figure
    :param max_health_val: highest health value on figure, barring thresholds.
    :type max_health_val: float
    :return: updated figure object
    :rtype: go.Figure
    """
    # get threshold values
    warn_val = st.session_state["health_warn_val"]
    alarm_val = st.session_state["health_alarm_val"]
    # check if threshold values are playing nice
    if warn_val >= alarm_val:
        st.error("Warning Threshold must be less than Alarm Threshold!")
        st.stop()
    # compute Y Axis upper limit
    ylim = max(alarm_val + 0.5, max_health_val)
    # add threshold lines
    fig.add_hline(
        y=warn_val,
        line_width=4,
        line_dash="dash",
        line_color=DEFAULT_WARNING_COLOR,
    )
    fig.add_hline(
        y=alarm_val,
        line_width=4,
        line_dash="dash",
        line_color=DEFAULT_ALARM_COLOR,
    )
    # apply Y axis limits
    fig.update_yaxes(range=[0, ylim])

    return fig


# ----------------
# Features Tab
# ----------------


def plot_features(feature_df: pd.DataFrame):
    """Plots one or more charts for the features tab.

    :param health_df: dataframe with health values.
    :type health_df: pd.DataFrame
    """
    separate_charts = st.session_state["separate_feat_charts"]

    if separate_charts:
        plot_features_separate(feature_df)
    else:
        plot_features_single(feature_df)


def plot_features_single(feature_df: pd.DataFrame):
    """Plots single feature chart with all the selected features.

    :param feature_df: dataframe with feature values.
    :type feature_df: pd.DataFrame
    """
    # init figure
    fig = go.Figure()
    xvar_label = st.session_state["feat_x"]
    # plot selected features on figure
    for feat in st.session_state["feat_y"]:
        # compute trace
        trace = go.Scattergl(
            x=feature_df[xvar_label] if xvar_label != "Index" else None,
            y=feature_df[feat],
            name=feat,
            mode=chart_types[st.session_state["feat_charttype"]],
        )
        # add trace to figure
        fig.add_trace(trace)
    # common updates to figure
    fig = update_figure(fig)
    # any specific updates to figure
    fig.update_layout(
        title_x=0.5,
        title_text="Features",
        xaxis_title=xvar_label,
        yaxis_title="Value",
        showlegend=True,
    )
    # render figure in app
    st.plotly_chart(fig, use_container_width=True, config=chart_config)


def plot_features_separate(feature_df: pd.DataFrame):
    """Plots multiple feature charts, one for each selected feature.

    :param feature_df: dataframe with feature values.
    :type feature_df: pd.DataFrame
    """
    xvar_label = st.session_state["feat_x"]
    # plot selected features on figure
    for feat in st.session_state["feat_y"]:
        # init figure
        fig = go.Figure()
        # compute trace
        trace = go.Scattergl(
            x=feature_df[xvar_label] if xvar_label != "Index" else None,
            y=feature_df[feat],
            name=feat,
            mode=chart_types[st.session_state["feat_charttype"]],
        )
        # add trace to figure
        fig.add_trace(trace)
        # common updates to figure
        fig = update_figure(fig)
        # any specific updates to figure
        fig.update_layout(
            title_x=0.5,
            title_text=feat,
            xaxis_title=xvar_label,
            yaxis_title="Value",
            showlegend=False,
        )
        # render figure in app
        st.plotly_chart(fig, use_container_width=True, config=chart_config)


def display_feature_table(df: pd.DataFrame):
    """displays a table for the provided dataframe, upto to a certain number of rows.
    Highlights the maximum value in a column.

    :param df: dataframe with feature data
    :type df: pd.DataFrame
    """
    feat_table_data = df
    total_row_count = df.shape[0]
    if total_row_count > FEAT_TABLE_ROW_LIMIT:
        st.warning(f"Limiting table to the first {FEAT_TABLE_ROW_LIMIT} rows.")
        feat_table_data = df.head(FEAT_TABLE_ROW_LIMIT)
    st.dataframe(
        feat_table_data,
        # feat_table_data.style.highlight_max(
        #     axis="index",
        #     color=FEAT_TABLE_HIGHLIGHT_COLOR,
        # ),
        use_container_width=True,
        height=500,
    )


# ----------------
# Raw Data Tab
# ----------------


def plot_rawdata(
    rawdata_group: h5py.Group,
    outer_looper: list[str | int],
    inner_looper: list[int | str],
    trace_labels: list[str],
    title_labels: list[str],
    chart_by_var: bool,
):
    """Plots raw data charts.
    This could end up several ways depending on the configured controls:
    1. one or more charts, for each selected record.
    2. one or more charts, for each selected data variable.

    :param rawdata_group: HDF5 Group containing raw data for the configured task.
    :type rawdata_group: h5py.Group
    :param outer_looper: list of Y Axis variables (or record indices).
    Corresponds to the number of charts.
    :type outer_looper: list[str | int]
    :param inner_looper: list of record indices (or Y Axis variables).
    Corresponds to the number of traces on each chart.
    :type inner_looper: list[int | str]
    :param trace_labels: labels for the traces, to display in the legend.
    list of record names (or Y Axis variables).
    :type trace_labels: list[str]
    :param title_labels: labels for the titles.
    list of Y Axis variables (or record names).
    :type title_labels: list[str]
    :param chart_by_var: truth value used to navigate our way through this puzzle.
    :type chart_by_var: bool
    """
    xvar_label = st.session_state["rawdata_x"]
    # loop over each chart
    for (outer_idx, outer_var) in enumerate(outer_looper):
        # init figure
        fig = go.Figure()
        # loop over each trace in chart
        for (inner_idx, inner_var) in enumerate(inner_looper):
            yvar_label = outer_var if chart_by_var else inner_var
            record_idx = inner_var if chart_by_var else outer_var
            xdata = (
                rawdata_group[xvar_label][record_idx] if xvar_label != "Index" else None
            )
            ydata = rawdata_group[yvar_label][record_idx]
            trace = go.Scattergl(
                x=xdata,
                y=ydata,
                name=trace_labels[inner_idx],
                mode=chart_types[st.session_state["rawdata_charttype"]],
            )
            # add trace to figure
            fig.add_trace(trace)

        # common updates to figure
        fig = update_figure(fig)
        # any specific updates to figure
        fig.update_layout(
            title_x=0.5,
            title_text=title_labels[outer_idx],
            xaxis_title=xvar_label,
            yaxis_title="Value",
            xaxis_hoverformat=".4g",
            showlegend=True,
        )
        # render figure in app
        st.plotly_chart(fig, use_container_width=True, config=chart_config)
