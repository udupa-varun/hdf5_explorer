import re

import streamlit as st

from . import plotting

# ----------------
# Generic/Common
# ----------------

# keywords that are used to detect datetime labels
VALID_TIME_LABELS: list = ["time", "timestamp", "date"]
REGEX_TS_PATTERNS: list[re.Pattern] = [
    re.compile(f"^{label}") for label in VALID_TIME_LABELS
]
# add PDX DAQ timestamp labels
REGEX_TS_PATTERNS.append(re.compile(r"^ts \([\w\s/]+\)$"))


def search_for_timestamp_in(options: list[str]) -> int | None:
    """searches provided list of strings for ones that match common
    timestamp patterns, and returns index for the first match.
    Returns None if no matches are found.

    :param options: list of strings to search in
    :type options: list[str]
    :return: index for the first match, or None (if no matches are found)
    :rtype: int | None
    """
    # loop over compiled regex patterns
    for pattern in REGEX_TS_PATTERNS:
        matches = [pattern.match(option.lower()) for option in options]
        # return the first match found
        if any(matches):
            match_idx = [i for (i, m) in enumerate(matches) if m is not None][0]
            return match_idx

    # if no matches were found, return None
    return None


def render_xy_controls(
    form_prefix: str,
    options: list[str],
    x_idx: int = -1,
    x_label: str = "X Axis",
    y_idx: int | None = 1,
    y_label: str = "Y Axis",
):
    """Renders common form controls like variable and chart type selection.

    :param form_prefix: prefix for the form name (form names must be unique in app)
    :type form_prefix: str
    :param options: list of variables to populate X and Y Axis controls with.
    An "Index" entry will be added to The X Axis options supplied.
    :type options: list[str]
    :param x_idx: default index to use when populating X Axis,
    defaults to -1. This is to account for the "Index" option added.
    :type x_idx: int, optional
    :param y_idx: default index to use when populating Y Axis,
    defaults to 1. Can be None to indicate empty selection.
    :type y_idx: int | None, optional
    """
    col_x, col_y, col_chart = st.columns([4, 4, 2])
    # search for timestamp patterns in variable options
    ts_idx = search_for_timestamp_in(options)
    # override X Axis if a timestamp variable was found
    if ts_idx is not None:
        x_idx = ts_idx

    xoptions = options.copy()
    xoptions.insert(0, "Index")

    with col_x:
        st.selectbox(
            label=f"{x_label}:",
            options=xoptions,
            key=f"{form_prefix}_x",
            index=x_idx + 1,
        )
    with col_y:
        st.multiselect(
            label=f"{y_label}:",
            options=options,
            key=f"{form_prefix}_y",
            default=options[y_idx] if y_idx is not None else None,
        )
    with col_chart:
        st.selectbox(
            label="Chart Type:",
            options=list(plotting.chart_types.keys()),
            index=0,
            key=f"{form_prefix}_charttype",
        )


# ----------------
# Health Tab
# ----------------


def render_health_controls(options: list[str], contrib_options: list[str]):
    """Renders form controls for the health tab.

    :param options: list of health component labels in the selected task.
    :type options: list[str]
    """
    with st.form("health_form"):
        (col_component, col_contrib, col_chart) = st.columns([4, 4, 2])
        with col_component:
            st.multiselect(
                label="Select Component(s):",
                options=options,
                default=options[0],
                key="health_components",
            )
        with col_contrib:
            st.multiselect(
                label="Select Contribution(s):",
                options=contrib_options,
                default=contrib_options[0] if contrib_options else None,
                key="health_contributions",
                help="Related components must be selected for contributions to show.",
            )
        with col_chart:
            st.selectbox(
                label="Chart Type:",
                options=list(plotting.chart_types.keys()),
                index=1,
                key="health_charttype",
            )

        st.checkbox(
            label="Plot components in separate charts", key="separate_health_charts"
        )
        st.form_submit_button("Plot Data")
        st.write(
            """<style>
        [data-testid="stHorizontalBlock"] {
            align-items: center;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )


# ----------------
# Features Tab
# ----------------


def render_feature_controls(options: list[str]):
    """Renders form controls for the features tab.

    :param options: list of feature labels in the selected task.
    :type options: list[str]
    """
    with st.form("feat_form"):
        # set default X as timestamp,
        # set default Y as first available feature, or else Record Index
        render_xy_controls(
            form_prefix="feat",
            options=options,
            x_idx=0,
            y_idx=1 if len(options) > 1 else 0,
        )

        st.checkbox(label="Plot in separate charts", key="separate_feat_charts")
        st.form_submit_button("Plot Data")


# ----------------
# Raw Data Tab
# ----------------


def render_rawdata_controls(record_options: list[str], var_options: list[str]):
    """Renders form controls for the raw data tab.

    :param record_options: list of records available in the selected date range.
    :type record_options: list[str]
    :param var_options: list of raw data labels in the selected task.
    :type var_options: list[str]
    """
    with st.form("rawdata_form"):
        (col_record, col_chartby) = st.columns([8, 2])
        with col_record:
            st.multiselect(
                label="Select Records:",
                options=record_options,
                key="rawdata_records",
                default=record_options[0],
            )
        with col_chartby:
            st.radio(
                label="Separate charts by:",
                options=["Record", "Variable"],
                horizontal=True,
                # label_visibility="hidden",
                key="rawdata_chartby",
            )
        # set default X as index,
        # set default Y as first available data variable
        render_xy_controls(
            form_prefix="rawdata1",
            options=var_options,
            x_idx=-1,
            y_idx=1 if len(var_options) > 1 else 0,
        )
        render_xy_controls(
            form_prefix="rawdata2",
            options=var_options,
            x_idx=-1,
            y_idx=None,
            x_label="X Axis 2",
            y_label="Y Axis 2",
        )

        st.form_submit_button("Plot Data")
