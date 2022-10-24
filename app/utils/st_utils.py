from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import streamlit as st

from . import h5_utils, plotting
from .st_forms import (
    render_health_controls,
    render_feature_controls,
    render_rawdata_controls,
)

# expected attributes in H5 groups representing DAQ Tasks
TASK_ATTRS = ["ancestors", "asset_type", "daq_task"]


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
    with h5_utils.read_file(file_path) as file_obj:
        task_names = [
            k
            for k in list(file_obj.keys())
            if isinstance(file_obj[k], h5py.Group)
        ]
        # task_names = list(file_obj.keys())

    return task_names


def render_dataset_controls():
    """Renders the controls for dataset configuration in the sidebar."""

    # with st.form("dataset_form"):
    # get directory path
    dir_path = st.text_input(
        label="Enter Directory Path:",
        value="./data",
        key="dir_path",
        # on_change=update_available_files
    )
    search_path = Path(dir_path).resolve()

    # stop if the directory does not exist
    if not search_path.is_dir():
        st.error("Invalid directory!", icon="🚨")
        st.stop()

    # get files from directory path
    file_paths: list[Path] = h5_utils.search_for_datafiles(search_path)

    # stop if there are no H5 files in the directory
    if not file_paths:
        st.warning("No files found.", icon="⚠️")
        st.stop()

    if "ready_to_tango" in st.session_state:
        del st.session_state["ready_to_tango"]

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
            st.error(
                "No groups were found in the data file. This is probably not a PDX H5 file!",
                icon="⚠️",
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
                st.error(
                    "Selected group does not have the expected DAQ Task attributes. "
                    "This is probably not a PDX H5 file!"
                )
                st.stop()

            timestamp_dset = file_obj[selected_task]["timestamps"]
            ts_min = datetime.fromtimestamp(timestamp_dset[0], tz=timezone.utc)
            ts_max = datetime.fromtimestamp(
                timestamp_dset[-1], tz=timezone.utc
            )

        # set date pickers based on timestamp range from file
        col1, col2 = st.columns(2)
        with col1:
            date_begin = st.date_input(
                "Start Date:",
                value=max(ts_max.date() - timedelta(days=30), ts_min.date()),
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
        # store as datetime in session state
        datetime_begin = datetime.combine(date_begin, datetime.min.time())
        st.session_state["datetime_begin"] = datetime_begin
        datetime_end = datetime.combine(date_end, datetime.min.time())
        st.session_state["datetime_end"] = datetime_end


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
        chunk_end_idx = h5_utils.get_closest_index_after_value(
            timestamp_dset, ts_end
        )
        chunk_begin_idx = h5_utils.get_closest_index_before_value(
            timestamp_dset, ts_begin
        )

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
    # update state first and foremost
    update_chunk_state()

    if (
        "ready_to_tango" in st.session_state
        and st.session_state["ready_to_tango"]
    ):
        file_path = Path(st.session_state["file_path_selected"])
        task: str = st.session_state["task"]

        # read from H5 file
        file_obj: h5py.File = h5_utils.read_file(file_path)
        st.session_state["file_path_loaded"] = Path(
            file_obj.filename
        ).resolve()

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

        # Health Tab
        with tab_health:
            health_group: h5py.Group = file_obj[task]["health"]
            health_options: list[str] = list(health_group.keys())
            # render plot controls
            render_health_controls(options=health_options)

            # prepare dataframe
            health_df = pd.DataFrame()
            for health_component in st.session_state["health_components"]:
                health_df[health_component] = health_group[health_component][
                    "health_values"
                ][chunk_begin_idx:chunk_end_idx]

            plotting.plot_health(health_df, datetime_chunk)

        # Features Tab
        with tab_features:
            # prepare dataframe based on configured chunk
            feat_dset: h5py.Dataset = file_obj[task]["features"]
            feature_names = feat_dset.attrs["names"]
            feature_chunk = feat_dset[chunk_begin_idx:chunk_end_idx]
            feature_df = pd.DataFrame(feature_chunk, columns=feature_names)
            feature_df.insert(0, column="Timestamp", value=datetime_chunk)

            # set up sub-tabs
            subtab_charts, subtab_table = st.tabs(["Charts", "Table"])
            with subtab_charts:
                # render plot controls
                render_feature_controls(options=feature_df.columns)

                plotting.plot_features(feature_df)
            with subtab_table:
                st.dataframe(
                    feature_df,
                    # feature_df.style.highlight_max(axis=0, color="red"),
                    use_container_width=True,
                )

        # Raw Data Tab
        with tab_rawdata:
            # prepare variable and record labels
            rawdata_group: h5py.Group = file_obj[task]["raw_data"]
            rawdata_names: list[str] = rawdata_group.attrs["names"]
            record_names: list[str] = meta_df["record_name"]

            # render plot controls
            render_rawdata_controls(
                record_options=record_names,
                var_options=rawdata_names,
            )
            # get indices for record names
            record_indices_in_file: list[int] = get_record_indices(
                record_names
            )

            # set up context for plotting chart(s)
            selected_record_names: list[str] = st.session_state[
                "rawdata_records"
            ]
            yvar_labels: list[str] = st.session_state["rawdata_y"]

            # hax
            chartby = st.session_state["rawdata_chartby"]
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
                st.error("Well this is unexpected...")
                st.stop()

            plotting.plot_rawdata(
                rawdata_group,
                outer_looper,
                inner_looper,
                trace_labels,
                title_labels,
                chart_by_var,
            )

        # Metadata Tab
        with tab_metadata:
            st.dataframe(meta_df, use_container_width=True)

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
