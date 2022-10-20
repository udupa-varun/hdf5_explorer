from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import streamlit as st

from utils import h5_utils, plotting, st_utils


def main():
    st.set_page_config(page_title="PDX HDF5 Explorer", layout="wide")
    st_utils.reduce_header_height()

    st.title("PDX H5 Explorer")

    # side bar
    with st.sidebar:
        st_utils.reduce_sidebar_height()

        st.title("Dataset")
        # get directory path
        dir_path = st.text_input(
            label="Enter Directory Path:",
            value="C:\code\python\PDX\data_store",
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

        selected_file_path: list[str] = []
        submitted: bool = False

        # display file names
        selected_file_path = st.selectbox(
            label="Select Data File:",
            options=file_paths,
            format_func=lambda x: x.relative_to(search_path),
            key="file_path",
        )
        st_utils.render_dataset_controls(selected_file_path)
        # st.session_state["ready_to_plot"] = False

        if st.session_state["FormSubmitter:dataset-form-Submit"]:
            file_path = Path(st.session_state["file_path"])
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
                st.session_state["ready_to_plot"] = True

        # st.session_state

    # main panel
    with st.container():
        # generate tabs
        (
            tab_health,
            tab_features,
            tab_rawdata,
            tab_metadata,
            tab_about,
        ) = st.tabs(["Health", "Features", "Raw Data", "Metadata", "About"])

        # needs stuff to be loaded in session
        if "ready_to_plot" in st.session_state and st.session_state["ready_to_plot"]:
            file_path = Path(st.session_state["file_path"])
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
            meta_df = st_utils.get_metadata_chunk(meta_dset)

            # Health Tab
            with tab_health:
                health_group: h5py.Group = file_obj[task]["health"]
                health_options: list[str] = list(health_group.keys())
                st_utils.render_health_controls(options=health_options)

                health_df = pd.DataFrame()
                for health_component in st.session_state["health_components"]:
                    health_df[health_component] = health_group[
                        health_component
                    ]["health_values"][chunk_begin_idx:chunk_end_idx]

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
                    st_utils.render_feature_controls(
                        options=feature_df.columns
                    )

                    plotting.plot_features(feature_df)
                with subtab_table:
                    # st.markdown("Features Table:")
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
                st_utils.render_rawdata_controls(
                    record_options=record_names,
                    var_options=rawdata_names,
                )
                # get indices for record names
                record_indices_in_file: list[
                    int
                ] = st_utils.get_record_indices(record_names)

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

            # About Tab
            with tab_about:
                readme_path = Path(__file__).parent.joinpath("about.md").resolve()
                with open(readme_path, "r") as f:
                    readme_data = f.read()
                st.markdown(readme_data)


main()
