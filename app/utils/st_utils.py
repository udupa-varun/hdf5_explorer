from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import streamlit as st

from . import h5_utils
from .plotting import chart_types


def reduce_header_height():
    """CSS hack to reduce height of the main panel."""
    reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
    st.markdown(reduce_header_height_style, unsafe_allow_html=True)


def reduce_sidebar_height():
    """CSS hack to reduce height of the sidebar."""
    reduce_sidebar_height_style = """
    <style>
        div.css-hxt7ib {padding-top:1rem;}
    </style>
    """
    st.markdown(reduce_sidebar_height_style, unsafe_allow_html=True)


@st.cache
def get_task_names(file_path: Path) -> list[str]:
    """Gets the names for tasks (HDF5 Groups) listed under a PDX HDF5 file.

    :param file_path: Path to the PDX HDF5 file.
    :type file_path: Path
    :return: list of task names
    :rtype: list[str]
    """
    task_names = []
    file_obj = h5_utils.read_file(file_path)
    task_names = list(file_obj.keys())

    return task_names


def render_dataset_controls(
    selected_file_path: Path,
):
    """Renders the form for dataset task and date range selection controls in the sidebar.

    :param selected_file_path: path to chosen HDF5 file
    :type selected_file_path: Path
    """
    with st.form("dataset-form") as dataset_form:

        if selected_file_path:
            task_names = get_task_names(selected_file_path)

            # file must have task groups present
            if not task_names:
                st.warning("No DAQ tasks present in file.", icon="⚠️")

            # show tasks
            selected_task = st.selectbox(
                label="Select DAQ Task:", options=task_names, key="task"
            )

            # compute timestamp range from file
            with h5_utils.read_file(selected_file_path) as file_obj:
                timestamp_dset = file_obj[selected_task]["timestamps"]
                ts_min = datetime.fromtimestamp(
                    timestamp_dset[0], tz=timezone.utc
                )
                ts_max = datetime.fromtimestamp(
                    timestamp_dset[-1], tz=timezone.utc
                )
            # set date pickers based on timestamp range from file
            col1, col2 = st.columns(2)
            with col1:
                date_begin = st.date_input(
                    "Start Date:",
                    value=max(
                        ts_max.date() - timedelta(days=30), ts_min.date()
                    ),
                    min_value=ts_min.date(),
                    max_value=ts_max.date(),
                )
            with col2:
                date_end = st.date_input(
                    "End Date:",
                    value=ts_max.date(),
                    min_value=ts_min.date(),
                    max_value=ts_max.date(),
                )

            st.session_state["datetime_begin"] = datetime.combine(
                date_begin, datetime.min.time()
            )
            st.session_state["datetime_end"] = datetime.combine(
                date_end, datetime.min.time()
            )

            submitted = st.form_submit_button("Submit")


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
            col_warn_color,
            col_alarm_val,
            col_alarm_color,
            _,
        ) = st.columns([2, 2, 1, 2, 1, 2], gap="medium")
        with col_warn_val:
            thresh_warn_val = st.number_input(
                label="Warning Threshold",
                min_value=0.0,
                # max_value=1.0,
                value=1.0,
                step=0.5,
                key="health_warn_val",
            )
        with col_warn_color:
            thresh_warn_color = st.color_picker(
                label="Warning Color",
                value="#FFA500",
                key="health_warn_color",
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
        with col_alarm_color:
            thresh_alarm_color = st.color_picker(
                label="Alarm Color",
                value="#FF0000",
                key="health_alarm_color",
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
            default_y_idx=2,
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
            default_y_idx=1,
        )

        submitted = st.form_submit_button("Plot Data")


def render_xy_controls(
    form_prefix: str,
    options: list[str],
    default_y_idx: int = 1,
):
    """Renders common form controls like variable and chart type selection.

    :param form_prefix: prefix for the form name (form names must be unique in app)
    :type form_prefix: str
    :param options: list of variables to populate X and Y Axis controls with.
    :type options: list[str]
    :param default_y_idx: default index to use when populating Y Axis (to avoid timestamps etc.),
    defaults to 1
    :type default_y_idx: int, optional
    """
    col_x, col_y, col_chart = st.columns([4, 4, 2])
    with col_x:
        x_var = st.selectbox(
            label="X Axis:",
            options=options,
            key=f"{form_prefix}_x",
        )
    with col_y:
        y_vars = st.multiselect(
            label="Y Axis:",
            options=options,
            key=f"{form_prefix}_y",
            default=options[default_y_idx],
        )
    with col_chart:
        selected_chart = st.selectbox(
            label="Chart Type:",
            options=list(chart_types.keys()),
            index=0,
            key=f"{form_prefix}_charttype",
        )


def get_metadata_chunk(meta_dset: h5py.Dataset) -> pd.DataFrame:
    """Computes and returns the relevant chunk of metadata
    for the configured task and date range.

    :param meta_dset: H5 Dataset for the selected task.
    :type meta_dset: h5py.Dataset
    :return: data frame containing metadata.
    :rtype: pd.DataFrame
    """
    meta_formatted = []
    # decode metadata for the configured chunk
    # stored as byte strings in file
    for row in meta_dset[
        st.session_state["chunk_begin_idx"] : st.session_state["chunk_end_idx"]
    ]:
        meta_formatted.append(np.array([s.decode() for s in row]))
    # create dataframe for display
    meta_df = pd.DataFrame(
        meta_formatted,
        columns=meta_dset.attrs["names"],
    )
    return meta_df


def get_record_indices(record_names: list[str]) -> list[int]:
    """Computes and returns indices for record names
    chosen from the configured chunk (relative to the HDF5 file).

    :param record_names: list of record names.
    :type record_names: list[str]
    :return: _description_
    :rtype: list[int]
    """
    # get configured chunk limits
    chunk_begin_idx: int = st.session_state["chunk_begin_idx"]
    chunk_end_idx: int = st.session_state["chunk_end_idx"]
    # get record indices (relative to this chunk)
    record_idx_map: dict = {k: i for i, k in enumerate(record_names)}
    # inter = set(record_names).intersection(st.session_state["rawdata_records"])
    record_indices_in_chunk = [
        record_idx_map[r] for r in st.session_state["rawdata_records"]
    ]
    # now get the actual file indices by using the chunk indices
    chunk_indices_in_file = list(
        range(
            chunk_begin_idx,
            chunk_end_idx,
        )
    )
    record_indices_in_file = [
        chunk_indices_in_file[i] for i in record_indices_in_chunk
    ]
    return record_indices_in_file
