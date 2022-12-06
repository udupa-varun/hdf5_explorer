from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import streamlit as st

from . import h5_utils, plotting
from .st_alerts import display_st_error, display_st_warning
from .st_forms import (
    render_feature_controls,
    render_health_controls,
    render_rawdata_controls,
)

# expected attributes in H5 groups representing DAQ Tasks
TASK_ATTRS = ["ancestors", "asset_type", "daq_task"]


def reduce_header_height():
    """CSS hack to reduce height of the main panel."""
    reduce_header_height_style = """
    <style>
        div.css-1vq4p4l {padding-top:4rem;}
    </style>
    """
    st.markdown(reduce_header_height_style, unsafe_allow_html=True)


def reduce_sidebar_height():
    """CSS hack to reduce height of the sidebar."""
    reduce_sidebar_height_style = """
    <style>
        div.css-18e3th9 {padding-top:1rem;}
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
    with h5_utils.read_file(file_path) as file_obj:
        task_names = [
            k for k in list(file_obj.keys()) if isinstance(file_obj[k], h5py.Group)
        ]

    return task_names


def render_dataset_controls():
    """Renders the controls for dataset configuration in the sidebar."""

    # get directory path
    dir_path = st.text_input(
        label="Search Directory Path:",
        value="./data",
        key="dir_path",
        # disabled=True,
    )
    search_path = Path(dir_path).resolve()

    # if the directory does not exist, create one
    if not search_path.is_dir():
        display_st_error("Directory does not exist.")
        st.stop()

    # get files from directory path
    file_paths: list[Path] = h5_utils.search_for_datafiles(search_path)

    # stop if there are no H5 files in the directory
    if not file_paths:
        display_st_warning("No files found.")
        st.stop()

    # display file names
    selected_file_path = st.selectbox(
        label="Select Data File:",
        options=file_paths,
        format_func=lambda file_path: file_path.relative_to(search_path),
        key="file_path_selected",
    )
    if selected_file_path.is_file():
        task_names = get_task_names(selected_file_path)

        # file must have groups present
        if not task_names:
            display_st_error(
                "No groups were found in the data file. "
                "This probably isn't a valid PDX H5 file!",
            )
            st.stop()

        # show tasks
        selected_task = st.selectbox(
            label="Select DAQ Task:", options=task_names, key="task"
        )

        # compute timestamp range from selected task
        with h5_utils.read_file(selected_file_path) as file_obj:
            # the selected task must have valid task attributes
            if list(file_obj[selected_task].attrs.keys()) != TASK_ATTRS:
                display_st_error(
                    "Selected group does not have the expected DAQ Task attributes. "
                    "This probably isn't a valid PDX DAQ task!",
                )
                st.stop()

            # get timestamp extremities for this task
            timestamp_dset = file_obj[selected_task]["timestamps"]
            ts_min = datetime.fromtimestamp(timestamp_dset[0], tz=timezone.utc)
            ts_max = datetime.fromtimestamp(timestamp_dset[-1], tz=timezone.utc)

            # determine date picker defaults
            # default end date is the last available date
            date_end_default = ts_max.date()

            # default start date is the most recent available date between:
            # 1. 30 days before the last available date (offset by duration)
            # 2. 1000 records before the last available date (offset by records)
            # offset by duration
            date_offset_by_duration = ts_max.date() - timedelta(days=30)
            # offset by number of records (limited by number of records available)
            num_records_available = timestamp_dset.shape[0]
            num_records_to_offset_by = min(num_records_available, 1000)
            date_offset_by_records = datetime.fromtimestamp(
                timestamp_dset[-1 * num_records_to_offset_by], tz=timezone.utc
            ).date()
            # select the most recent of the two as the default start date
            date_begin_default = max(date_offset_by_duration, date_offset_by_records)

        # set date pickers based on timestamp range from file
        col1, col2 = st.columns(2)
        with col1:
            date_begin = st.date_input(
                "Start Date:",
                value=date_begin_default,
                min_value=ts_min.date(),
                max_value=ts_max.date(),
            )
        with col2:
            date_end = st.date_input(
                "End Date:",
                value=date_end_default,
                min_value=ts_min.date(),
                max_value=ts_max.date(),
            )
        # store as datetime in session state
        datetime_begin = datetime.combine(
            date=date_begin, time=datetime.min.time(), tzinfo=timezone.utc
        )
        st.session_state["datetime_begin"] = datetime_begin
        datetime_end = datetime.combine(
            date=date_end, time=datetime.min.time(), tzinfo=timezone.utc
        )
        st.session_state["datetime_end"] = datetime_end

        # update chunk indices based on selected date range
        update_chunk_state()
        if "chunk_begin_idx" in st.session_state:
            num_records_in_selected_range = (
                st.session_state["chunk_end_idx"] - st.session_state["chunk_begin_idx"]
            )
            st.caption(
                f"Found {num_records_in_selected_range} "
                "record(s) in the selected date range."
            )


def update_chunk_state():
    """updates parts of the session state that are required to plot data."""
    file_path = Path(st.session_state["file_path_selected"])
    task: str = st.session_state["task"]

    # read from H5 file
    with h5_utils.read_file(file_path) as file_obj:
        # get timestamp dataset
        timestamp_dset: h5py.Dataset = file_obj[task]["timestamps"]

        # get chunk limits
        ts_begin = st.session_state["datetime_begin"].timestamp()
        ts_end = st.session_state["datetime_end"].timestamp()
        # get first record after end date
        chunk_end_idx = h5_utils.get_closest_index_on_or_after_value(
            dset=timestamp_dset, search_val=ts_end, pref="after"
        )
        # get first record after start date
        chunk_begin_idx = h5_utils.get_closest_index_on_or_after_value(
            dset=timestamp_dset, search_val=ts_begin, pref="on"
        )

        # handle edge cases
        # if end index is 0 (single record present)
        if chunk_end_idx == 0:
            chunk_end_idx = 1
        # if no matches were found
        # should only encounter this if there are issues with the timestamp data
        if chunk_end_idx is None or chunk_begin_idx is None:
            display_st_error("No records were found!")
            st.stop()

        # store chunk info in session
        st.session_state["chunk_begin_idx"] = chunk_begin_idx
        st.session_state["chunk_end_idx"] = chunk_end_idx
    st.session_state["ready_to_tango"] = True


def update_main_panel():
    """logic for updating the main panel."""
    # generate tabs
    (
        tab_health,
        tab_features,
        tab_rawdata,
        tab_metadata,
        # tab_about,
    ) = st.tabs(
        [
            "Health",
            "Features",
            "Raw Data",
            "Metadata",
            # "About",
        ]
    )
    if "ready_to_tango" in st.session_state and st.session_state["ready_to_tango"]:
        file_path = Path(st.session_state["file_path_selected"])
        task: str = st.session_state["task"]

        # read from H5 file
        file_obj: h5py.File = h5_utils.read_file(file_path)

        # get timestamp dataset
        timestamp_dset: h5py.Dataset = file_obj[task]["timestamps"]

        # get configured chunk limits
        chunk_begin_idx = st.session_state["chunk_begin_idx"]
        chunk_end_idx = st.session_state["chunk_end_idx"]

        # get datetime chunk
        datetime_chunk: np.ndarray = np.array(
            [
                datetime.fromtimestamp(t, tz=timezone.utc)
                for t in timestamp_dset[chunk_begin_idx:chunk_end_idx]
            ]
        )
        # get metadata dataset (used in both rawdata and metadata tabs)
        meta_dset: h5py.Dataset = file_obj[task]["record_meta_data"]
        meta_df = get_metadata_chunk(meta_dset)

        # stop if there are no records in the configured range
        if meta_df.empty:
            display_st_error("No records in the selected range!")
            st.stop()

        # Health Tab
        with tab_health:
            # set up reference to task health
            health_group: h5py.Group = file_obj[task]["health"]
            health_component_names = h5_utils.get_group_members(health_group)

            # only proceed if health components are present
            if health_component_names:
                # store contribution data
                contrib_df = get_contribution_data(health_group, health_component_names)

                # render plot controls
                render_health_controls(
                    options=health_component_names,
                    contrib_options=list(contrib_df.columns),
                )

                # store health values and thresholds for selected components
                (health_df, thresh_store) = get_health_data(health_group)

                # render charts
                plotting.plot_health(
                    health_df, contrib_df, thresh_store, datetime_chunk
                )
            else:
                display_st_warning("No health components detected.")

        # Features Tab
        with tab_features:
            # prepare dataframe based on configured chunk
            feat_dset: h5py.Dataset = file_obj[task]["features"]
            feature_names: np.ndarray = h5_utils.get_obj_attribute(feat_dset, "names")
            feature_chunk: np.ndarray = feat_dset[chunk_begin_idx:chunk_end_idx]

            # if features are present, load them into dataframe
            if feature_names.size > 0 and feature_chunk.size > 0:
                feature_df = pd.DataFrame(feature_chunk, columns=feature_names)
                feature_df.insert(0, column="Timestamp", value=datetime_chunk)
            # otherwise init empty dataframe
            else:
                feature_df = pd.DataFrame()

            # only proceed if feature values are available
            if not feature_df.empty:
                # set up sub-tabs
                subtab_charts, subtab_table = st.tabs(["Charts", "Table"])
                with subtab_charts:
                    # render plot controls
                    render_feature_controls(options=list(feature_df.columns))

                    # render charts
                    plotting.plot_features(feature_df)
                with subtab_table:
                    # render feature table
                    plotting.display_feature_table(feature_df)
            else:
                display_st_warning("No features detected.")

        # Raw Data Tab
        with tab_rawdata:
            # prepare variable and record labels
            rawdata_group: h5py.Group = file_obj[task]["raw_data"]
            rawdata_var_names = h5_utils.get_group_members(rawdata_group)
            record_names: np.ndarray = meta_df["record_name"]

            # check if raw data variables are available
            if rawdata_var_names:
                # prune for numeric dtypes
                rawdata_var_dtypes = [rawdata_group[k].dtype for k in rawdata_var_names]
                rawdata_var_names = [
                    var_name
                    for (var_name, var_dtype) in zip(
                        rawdata_var_names, rawdata_var_dtypes
                    )
                    if np.issubdtype(var_dtype, np.number)
                ]

            # check if raw data variables are still available after pruning
            if rawdata_var_names:
                # render plot controls
                render_rawdata_controls(
                    record_options=list(record_names),
                    var_options=rawdata_var_names,
                )

                # get indices for record names
                record_indices_in_file: list[int] = get_record_indices(record_names)

                # set up context for plotting charts
                selected_record_names: list[str] = st.session_state["rawdata_records"]
                chartby = st.session_state["rawdata_chartby"]

                # plot X1, Y1
                plotting.plot_rawdata_block(
                    rawdata_group,
                    chartby,
                    record_indices_in_file,
                    selected_record_names,
                    chart_block=1,
                )
                # plot X2, Y2
                plotting.plot_rawdata_block(
                    rawdata_group,
                    chartby,
                    record_indices_in_file,
                    selected_record_names,
                    chart_block=2,
                )
            else:
                display_st_warning("No raw data available.")

        # Metadata Tab
        with tab_metadata:
            st.dataframe(meta_df, use_container_width=True, height=500)

        # close file handler
        file_obj.close()

    # About Tab
    # static content, so it can be rendered without submitting dataset form
    # with tab_about:
    #     readme_path = Path(__file__).parents[1].joinpath("about.md").resolve()
    #     with open(readme_path, "r") as f:
    #         readme_data = f.read()
    #     st.markdown(readme_data)


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
    # reverse the order of records metadata (most recent first)
    # note that this in turn affects the order in which raw data records are shown
    meta_formatted = meta_formatted[::-1]
    # create dataframe for display
    meta_df = pd.DataFrame(
        meta_formatted,
        columns=h5_utils.get_obj_attribute(meta_dset, "names"),
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
    record_indices_in_file = [chunk_indices_in_file[i] for i in record_indices_in_chunk]
    return record_indices_in_file


def get_contribution_data(_health_group, health_component_names) -> pd.DataFrame:
    """gets contribution data from file for specified health components.
    Contribution labels are modified to indicate the health component they belong to.

    :param _health_group: health group for a given task
    :type _health_group: h5py.Group
    :param health_component_names: list of health component labels
    :type health_component_names: list[str]
    :return: contribution data
    :rtype: pd.DataFrame
    """
    # init dataframe for contributions
    contrib_df = pd.DataFrame()
    # loop over health components present in selected file
    for health_component in health_component_names:
        # check if this component has contributions present
        if "contributions" in _health_group[health_component].keys():
            # get contribution labels from file
            component_contrib_labels = list(
                h5_utils.get_obj_attribute(
                    _health_group[health_component]["contributions"],
                    "names",
                )
            )
            # hack that adds component name to contribution label
            component_contrib_labels = [
                f"{health_component}|{c}" for c in component_contrib_labels
            ]

            # get contribution data for this component
            contrib_chunk = _health_group[health_component]["contributions"][
                st.session_state["chunk_begin_idx"] : st.session_state["chunk_end_idx"]
            ]
            contrib_chunk_df = pd.DataFrame(
                contrib_chunk,
                columns=component_contrib_labels,
            )
            # add contributions to the dataframe
            contrib_df = pd.concat([contrib_df, contrib_chunk_df], axis=1)
    return contrib_df


def get_health_data(_health_group) -> tuple[pd.DataFrame, list]:
    """gets health values and threshold values for specified health group.
    The health components chosen are the ones present in the session state.

    :param _health_group: health group for a given task
    :type _health_group: h5py.Group
    :return: health values and thresholds
    :rtype: tuple[pd.DataFrame, list]
    """
    # init dataframe for health values
    health_df = pd.DataFrame()
    thresh_store = {}
    # loop over selected health components
    selected_health_components = st.session_state["health_components"]
    for health_component in selected_health_components:
        # store health values
        health_df[health_component] = _health_group[health_component]["health_values"][
            st.session_state["chunk_begin_idx"] : st.session_state["chunk_end_idx"]
        ]
        # store health thresholds
        threshold_values = h5_utils.get_obj_attribute(
            _health_group[health_component], "thresholds"
        )
        # pull out the warning and alarm threshold values, if available
        if threshold_values is not None:
            threshold_values = list(threshold_values)[1:3]
        # otherwise, store with default values
        else:
            # TODO: use defaults from environment vars?
            threshold_values = [1.0, 2.0]
        thresh_store[health_component] = threshold_values

    return (health_df, thresh_store)
