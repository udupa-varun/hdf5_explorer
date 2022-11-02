import streamlit as st

from . import plotting


def render_health_controls(options: list[str]):
    """Renders form controls for the health tab.

    :param options: list of health component labels in the selected task.
    :type options: list[str]
    """
    with st.form("health_form"):
        selected_component = st.multiselect(
            label="Select Component(s):",
            options=options,
            default=options[0],
            key="health_components",
        )
        (
            col_separate,
            col_warn_val,
            col_alarm_val,
            _,
        ) = st.columns([2, 2, 2, 4], gap="medium")
        with col_warn_val:
            thresh_warn_val = st.number_input(
                label="Warning Threshold",
                min_value=0.0,
                # max_value=1.0,
                value=1.0,
                step=0.5,
                key="health_warn_val",
            )
        with col_alarm_val:
            thresh_alarm_val = st.number_input(
                label="Alarm Threshold",
                min_value=0.0,
                # max_value=2.0,
                value=2.0,
                step=0.5,
                key="health_alarm_val",
            )
        with col_separate:
            separate_health_charts: bool = st.checkbox(
                label="Plot in separate charts", key="separate_health_charts"
            )
            submitted = st.form_submit_button("Plot Data")
        st.write(
            """<style>
        [data-testid="stHorizontalBlock"] {
            align-items: center;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )


def render_feature_controls(options: list[str]):
    """Renders form controls for the features tab.

    :param options: list of feature labels in the selected task.
    :type options: list[str]
    """
    with st.form("feat_form") as xy_form:
        render_xy_controls(
            form_prefix="feat",
            options=options,
            x_idx=1,
            y_idx=2,
        )

        separate_feat_charts: bool = st.checkbox(
            label="Plot in separate charts", key="separate_feat_charts"
        )
        submitted = st.form_submit_button("Plot Data")


def render_rawdata_controls(record_options: list[str], var_options: list[str]):
    """Renders form controls for the raw data tab.

    :param record_options: list of records available in the selected date range.
    :type record_options: list[str]
    :param var_options: list of raw data labels in the selected task.
    :type var_options: list[str]
    """
    with st.form("rawdata_form") as xy_form:
        (col_record, col_chartby) = st.columns([8, 2])
        with col_record:
            selected_records = st.multiselect(
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

        render_xy_controls(
            form_prefix="rawdata",
            options=var_options,
            x_idx=0,
            y_idx=1,
        )

        submitted = st.form_submit_button("Plot Data")


def render_xy_controls(
    form_prefix: str,
    options: list[str],
    x_idx: int = 0,
    y_idx: int = 1,
):
    """Renders common form controls like variable and chart type selection.

    :param form_prefix: prefix for the form name (form names must be unique in app)
    :type form_prefix: str
    :param options: list of variables to populate X and Y Axis controls with.
    :type options: list[str]
    :param x_idx: default index to use when populating X Axis,
    defaults to 0. Note that an "Index" option is added here and must be accounted for.
    :type x_idx: int, optional
    :param y_idx: default index to use when populating Y Axis,
    defaults to 1
    :type y_idx: int, optional
    """
    col_x, col_y, col_chart = st.columns([4, 4, 2])
    xoptions = options.copy()
    xoptions.insert(0, "Index")
    with col_x:
        x_var = st.selectbox(
            label="X Axis:",
            options=xoptions,
            key=f"{form_prefix}_x",
            index=x_idx,
        )
    with col_y:
        y_vars = st.multiselect(
            label="Y Axis:",
            options=options,
            key=f"{form_prefix}_y",
            default=options[y_idx],
        )
    with col_chart:
        selected_chart = st.selectbox(
            label="Chart Type:",
            options=list(plotting.chart_types.keys()),
            index=0,
            key=f"{form_prefix}_charttype",
        )
