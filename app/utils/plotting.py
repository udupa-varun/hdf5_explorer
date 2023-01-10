import h5py
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .st_alerts import display_st_error, display_st_info, display_st_warning

# ----------------
# Generic/Common
# ----------------

CHART_HEIGHT = 450
PLOTLY_COLORS = plotly.colors.qualitative.Plotly

# max number of markers per chart
# if exceeded, figure will force line charts
MAX_MARKERS_PER_FIGURE = int(1e6)

# colors for health thresholds
THRESH_WARNING_COLOR = "#FFA500"
THRESH_ALARM_COLOR = "#FF0000"

# max number of allowable rows for the features table
FEAT_TABLE_ROW_LIMIT = int(1e3)
FEAT_TABLE_HIGHLIGHT_COLOR = "#FF4B4B"

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


def apply_generic_figure_updates(fig: go.Figure) -> go.Figure:
    """Layout updates to be applied to ALL figures generated.
    Any updates specific to a tab do not belong here.

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
            xanchor="left",
            y=-0.5,
            x=0,
        ),
        hovermode="x unified",
    )
    # updates for X axis
    fig.update_xaxes(
        showline=True,
        mirror=True,
        linewidth=1,
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        showticklabels=True,
    )
    # updates for Y axis
    fig.update_yaxes(
        showline=True,
        mirror=True,
        linewidth=1,
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        showticklabels=True,
        title="Value",
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
    # exit if figure has no data
    if not fig["data"]:
        return fig

    num_data_points = get_data_point_count(fig)

    # get chart type requested
    chart_mode = fig["data"][0]["mode"]

    # set line opacity
    fig.update_traces(opacity=0.7)

    # check if we are "marker-safe"
    if num_data_points > MAX_MARKERS_PER_FIGURE:
        # we are not marker-safe,
        # so override chart type with a translucent line plot
        # warning displays once per figure
        display_st_warning(
            f"Too many data points ({num_data_points})! Forcing a line chart.",
        )
        fig.update_traces(mode="lines")
        # # also disable hover labels
        # fig.update_layout(hovermode=False)
    # safe to add marker borders
    elif "markers" in chart_mode:
        fig.update_traces(
            opacity=0.8,  # override line opacity
            marker=dict(
                size=10,
                line=dict(width=1.5, color="DarkSlateGrey"),
                opacity=0.9,  # marker opacity
            ),
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


def plot_health(
    health_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    thresh_store: dict[str, list],
    datetime_chunk: np.ndarray,
):
    """Plots one or more charts for the health tab.

    :param health_df: dataframe with health values.
    :type health_df: pd.DataFrame
    :param contrib_df: dataframe with contribution values.
    :type contrib_df: pd.DataFrame
    :param thresh_store: threshold values for health components.
    :type thresh_store: dict[str, list]
    :param datetime_chunk: array of record timestamps
    :type datetime_chunk: np.ndarray
    """
    separate_charts = st.session_state["separate_health_charts"]

    if separate_charts:
        plot_health_separate(health_df, contrib_df, thresh_store, datetime_chunk)
    else:
        plot_health_single(health_df, contrib_df, thresh_store, datetime_chunk)


def plot_health_single(
    health_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    thresh_store: dict[str, list],
    datetime_chunk: np.ndarray,
):
    """Plots a single health chart with all the selected health components.

    :param health_df: dataframe with health values.
    :type health_df: pd.DataFrame
    :param contrib_df: dataframe with contribution values.
    :type contrib_df: pd.DataFrame
    :param thresh_store: threshold values for health components.
    :type thresh_store: dict[str, list]
    :param datetime_chunk: array of record timestamps
    :type datetime_chunk: np.ndarray
    """
    # init figure
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[3, 1],
        # subplot_titles=["Health Index", "Contributions"],
        # x_title="Timestamp",
        # vertical_spacing=0.4,
    )
    # loop over selected health components
    for (h_idx, component) in enumerate(health_df.columns):
        # compute trace
        h_trace = go.Scattergl(
            x=datetime_chunk,
            y=health_df[component],
            name=component,
            mode=chart_types[st.session_state["health_charttype"]],
            # line=dict(color=PLOTLY_COLORS[h_idx]),
        )
        # add trace to top subplot
        fig.add_trace(h_trace, row=1, col=1)

    selected_contribs = st.session_state["health_contributions"]
    # filter for relevant contributions
    relevant_contribs = [
        c for c in selected_contribs if c.split("|")[0] in health_df.columns
    ]
    # loop over selected health contributions
    for (c_idx, contrib) in enumerate(relevant_contribs):
        # compute trace
        c_trace = go.Scattergl(
            x=datetime_chunk,
            y=contrib_df[contrib],
            name=contrib,
            mode=chart_types[st.session_state["health_charttype"]],
            # line=dict(color=PLOTLY_COLORS[c_idx]),
        )
        # add trace to bottom subplot
        fig.add_trace(c_trace, row=2, col=1)

    # common updates to figure
    fig = apply_generic_figure_updates(fig)

    # specific updates to figure
    # if multiple components are selected, inform user how this is being handled
    if len(health_df.columns) > 1:
        display_st_info(
            """Thresholds shown belong to the first selected component.
            To change this, either plot the components separately
            or change the order in which they are selected.
            """,
        )
    # set up required params
    fig_params = {}
    fig_params["title_text"] = "Multiple Components"
    # get the max value across all health components (for setting ylim)
    fig_params["max_health_val"]: float = np.max(health_df.max())
    fig_params["min_health_val"]: float = np.min(health_df.min())
    # only get thresholds for the first selected component
    fig_params["threshold_values"] = thresh_store[health_df.columns[0]]
    # apply updates
    fig = apply_health_figure_updates(fig, fig_params)

    # render figure in app
    st.plotly_chart(fig, use_container_width=True, config=chart_config)
    # render line separator after figure
    st.markdown("""---""")


def plot_health_separate(
    health_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    thresh_store: dict[str, list],
    datetime_chunk: np.ndarray,
):
    """Plots multiple health charts, one for each selected health component.

    :param health_df: dataframe with health values.
    :type health_df: pd.DataFrame
    :param contrib_df: dataframe with contribution values.
    :type contrib_df: pd.DataFrame
    :param thresh_store: threshold values for health components.
    :type thresh_store: dict[str, list]
    :param datetime_chunk: array of record timestamps
    :type datetime_chunk: np.ndarray
    """
    for (h_idx, component) in enumerate(health_df.columns):
        # init figure
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[3, 1],
            # subplot_titles=["Health Index", "Contributions"],
            # x_title="Timestamp",
        )
        # compute trace
        h_trace = go.Scattergl(
            x=datetime_chunk,
            y=health_df[component],
            name=component,
            mode=chart_types[st.session_state["health_charttype"]],
            # line=dict(color=PLOTLY_COLORS[h_idx]),
        )
        # add trace to top subplot
        fig.add_trace(h_trace, row=1, col=1)

        # get selected contributions
        selected_contribs = st.session_state["health_contributions"]
        # filter for relevant contributions
        relevant_contribs = [
            c for c in selected_contribs if c.split("|")[0] == component
        ]
        # loop over relevant health contributions
        for (c_idx, contrib) in enumerate(relevant_contribs):
            # compute trace
            c_trace = go.Scattergl(
                x=datetime_chunk,
                y=contrib_df[contrib],
                name=contrib,
                mode=chart_types[st.session_state["health_charttype"]],
                # line=dict(color=PLOTLY_COLORS[c_idx]),
            )
            # add trace to bottom subplot
            fig.add_trace(c_trace, row=2, col=1)

        # TODO: if there are no relevant contributions, can the empty subplot be removed?

        # common updates to figure
        fig = apply_generic_figure_updates(fig)

        # specific updates to figure
        # set up required params
        fig_params = {}
        fig_params["title_text"] = component
        # get the max value for this health components
        fig_params["max_health_val"]: float = np.max(health_df[component])
        fig_params["min_health_val"]: float = np.min(health_df[component])
        # get thresholds for this component
        fig_params["threshold_values"] = thresh_store[component]
        # apply updates
        fig = apply_health_figure_updates(fig, fig_params)

        # render figure in app
        st.plotly_chart(fig, use_container_width=True, config=chart_config)
        # render line separator after figure
        st.markdown("""---""")


def apply_health_figure_updates(fig: go.Figure, fig_params: dict) -> go.Figure:
    """Performs health figure updates.

    :param fig: input figure object.
    :type fig: go.Figure
    :param fig_params: a collection of parameters to be used when modifying the figure.
    :type fig_params: dict
    :return: updated figure object.
    :rtype: go.Figure
    """
    # plot threshold lines on health subplot
    fig = plot_threshold_lines(
        fig, fig_params["threshold_values"], fig_params["max_health_val"], row=1
    )

    # apply Y axis limits for health, taking thresholds into account
    (ylim_lower, ylim_upper) = get_health_ylim(
        fig_params["min_health_val"],
        fig_params["max_health_val"],
        fig_params["threshold_values"],
    )
    fig.update_yaxes(
        range=[ylim_lower, ylim_upper],
        row=1,
    )

    # update Y axis limits for contributions
    fig.update_yaxes(
        range=[-0.1, 1],
        row=2,
    )

    # updates for anything other than axes
    fig.update_layout(
        title_x=0.5,
        title_text=fig_params["title_text"],
        showlegend=True,
        height=int(CHART_HEIGHT * 1.1),
    )
    # axis property overrides
    fig.update_layout(
        xaxis2_title="Timestamp",
        yaxis_title="Health Index",
        yaxis2_showgrid=False,  # special case for contrib
        yaxis2_title="Contributions",
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
            font_color=st.get_option("theme.textColor"),
            activecolor=st.get_option("theme.primaryColor"),
            bgcolor=st.get_option("theme.secondaryBackgroundColor"),
        ),
        row=1,
    )

    return fig


def plot_threshold_lines(
    fig: go.Figure,
    threshold_values: list[float],
    max_health_val: float,
    row: int = 1,
) -> go.Figure:
    """Plots horizontal lines on the given figure object,
    based on the configured threshold controls.

    :param fig: plotly figure object.
    :type fig: go.Figure
    :param row: row index for subplot in figure.
    :type row: int
    :return: updated figure object
    :rtype: go.Figure
    """
    # get threshold values
    warn_val = threshold_values[0]
    alarm_val = threshold_values[1]
    # check if threshold values are playing nice
    if warn_val >= alarm_val:
        display_st_error("Warning Threshold must be less than Alarm Threshold!")
        st.stop()
    # add threshold lines
    fig.add_hline(
        y=warn_val,
        line_width=2,
        # line_dash="dash",
        line_color=THRESH_WARNING_COLOR,
        row=row,
        # opacity=0.6,
    )
    fig.add_hline(
        y=alarm_val,
        line_width=2,
        # line_dash="dash",
        line_color=THRESH_ALARM_COLOR,
        row=row,
        # opacity=0.6,
    )

    return fig


def get_health_ylim(
    min_health_val: float, max_health_val: float, threshold_values: list[str]
) -> tuple[float]:
    """determines Y axis limits for the health index plot.

    :param min_health_val: minimum health value being plotted.
    :type min_health_val: float
    :param max_health_val: maximum health value being plotted.
    :type max_health_val: float
    :param threshold_values: threshold values being plotted.
    :type threshold_values: list[float]
    :return: lower and upper Y axis limits.
    :rtype: tuple[float]
    """
    ylim_lower: float = np.min([-0.1, min_health_val])
    ylim_upper: float = np.max([threshold_values[1], max_health_val]) + 0.5

    return (ylim_lower, ylim_upper)


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
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
    )
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
    fig = apply_generic_figure_updates(fig)
    # any specific updates to figure
    fig.update_layout(
        # title_x=0.5,
        title_text="Multiple Features",
        xaxis_title=xvar_label,
        showlegend=True,
        height=CHART_HEIGHT,
    )

    # render figure in app
    st.plotly_chart(fig, use_container_width=True, config=chart_config)
    # render line separator after figure
    st.markdown("""---""")


def plot_features_separate(feature_df: pd.DataFrame):
    """Plots multiple feature charts, one for each selected feature.

    :param feature_df: dataframe with feature values.
    :type feature_df: pd.DataFrame
    """

    xvar_label = st.session_state["feat_x"]
    yvar_labels = st.session_state["feat_y"]
    num_rows = len(yvar_labels)

    # init figure
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=yvar_labels,
    )
    # plot selected features on figure
    for (row_idx, feat) in enumerate(yvar_labels):
        # compute trace
        trace = go.Scattergl(
            x=feature_df[xvar_label] if xvar_label != "Index" else None,
            y=feature_df[feat],
            name=feat,
            mode=chart_types[st.session_state["feat_charttype"]],
            line=dict(color=PLOTLY_COLORS[0]),
        )
        # add trace to subplot
        fig.add_trace(trace, row=row_idx + 1, col=1)
    # common updates to figure
    fig = apply_generic_figure_updates(fig)
    # any specific updates to figure
    fig.update_layout(
        # title_x=0.5,
        title_text="Features",
        showlegend=False,
        height=CHART_HEIGHT * num_rows,
    )
    fig.update_xaxes(title=xvar_label)

    # render figure in app
    st.plotly_chart(fig, use_container_width=True, config=chart_config)
    # render line separator after figure
    st.markdown("""---""")


def display_feature_table(df: pd.DataFrame):
    """displays a table for the provided dataframe, upto to a certain number of rows.
    Highlights the maximum value in a column.

    :param df: dataframe with feature data
    :type df: pd.DataFrame
    """
    feat_table_data = df
    total_row_count = df.shape[0]
    if total_row_count > FEAT_TABLE_ROW_LIMIT:
        display_st_warning(f"Limiting table to the first {FEAT_TABLE_ROW_LIMIT} rows.")
        feat_table_data = df.head(FEAT_TABLE_ROW_LIMIT)
    st.dataframe(
        feat_table_data,
        use_container_width=True,
        height=500,
    )


# ----------------
# Raw Data Tab
# ----------------


def plot_rawdata_charts(
    rawdata_group: h5py.Group,
    xvar_label: str,
    super_title: str,
    chart_mode: str,
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
    :param xvar_label: X Axis variable label
    :type xvar_label: str
    :param super_title: title for the raw data block
    :type super_title: str
    :param chart_mode: chart type to use for this block
    :type chart_mode: str
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
    # generate at least one row
    num_rows = max(len(outer_looper), 1)
    # init figure
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=title_labels,
    )
    # loop over each chart
    for (outer_idx, outer_var) in enumerate(outer_looper):
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
                mode=chart_mode,
                line=dict(color=PLOTLY_COLORS[inner_idx]),
                # only show legends for the top subplot because
                # colors are same across subplots and
                # legend is specific to figure, not subplot
                showlegend=True,  # if outer_idx == 0 else False,
                legendgroup=f"chart{outer_idx}",
            )
            # add trace to figure
            fig.add_trace(
                trace,
                row=outer_idx + 1,
                col=1,
            )

    # common updates to figure
    fig = apply_generic_figure_updates(fig)

    # any specific updates to figure
    fig.update_layout(
        # title_x=0.5,
        title_text=super_title,
        xaxis_hoverformat=".4g",
        height=CHART_HEIGHT * num_rows,
        legend=dict(
            y=-0.4 / num_rows,
            groupclick="toggleitem",
        ),
    )
    fig.update_xaxes(title=xvar_label)

    # render figure in app
    st.plotly_chart(fig, use_container_width=True, config=chart_config)
    # render line separator after figure
    st.markdown("""---""")


def plot_rawdata_block(
    rawdata_group: h5py.Group,
    chartby: str,
    record_indices_in_file: list[int],
    selected_record_names: list[str],
    chart_block: int,
):
    """Sets up and plots a block of rawdata charts.

    :param rawdata_group: HDF5 Group containing raw data for the configured task.
    :type rawdata_group: h5py.Group
    :param chartby: session state data for chart by variable/record
    :type chartby: str
    :param record_indices_in_file: Indices for selected records in file.
    :type record_indices_in_file: list[int]
    :param selected_record_names: selected record names
    :type selected_record_names: list[str]
    :param chart_block: the chart block to plot in (1 for X1,Y1 or 2 for X2,Y2)
    :type chart_block: int
    """
    # set up params for chart generation
    xvar_label: str = st.session_state[f"rawdata{chart_block}_x"]
    yvar_labels: list[str] = st.session_state[f"rawdata{chart_block}_y"]
    chart_mode: str = chart_types[st.session_state[f"rawdata{chart_block}_charttype"]]
    super_title: str = f"Raw Data Group {chart_block}"
    # outer decides the number of subplots,
    # inner decides the number of traces in a subplot
    if chartby == "Variable":
        outer_looper = yvar_labels
        inner_looper = record_indices_in_file
        trace_labels = selected_record_names
        title_labels = yvar_labels
        chart_by_var: bool = True
    elif chartby == "Record":
        outer_looper = record_indices_in_file
        inner_looper = yvar_labels
        trace_labels = yvar_labels
        title_labels = selected_record_names
        chart_by_var: bool = False
    else:
        display_st_error(
            "Unexpected value received from raw data session state! Aborting..."
        )
        st.stop()

    # render charts for this block
    plot_rawdata_charts(
        rawdata_group,
        xvar_label,
        super_title,
        chart_mode,
        outer_looper,
        inner_looper,
        trace_labels,
        title_labels,
        chart_by_var,
    )
