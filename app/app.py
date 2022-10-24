import streamlit as st
import os
from pathlib import Path

from utils import st_utils

DEBUG: bool = os.getenv("PDX_H5_DEBUG", "False").lower() in ("true", "1", "t")


def main():
    """main driver for the app."""
    menu_items = get_menu_items()
    st.set_page_config(
        page_title="PDX HDF5 Explorer",
        layout="wide",
        menu_items=menu_items,
    )
    st_utils.reduce_header_height()
    st.title("PDX H5 Explorer")

    # side bar
    with st.sidebar:
        st_utils.reduce_sidebar_height()
        st.header("Dataset")
        st_utils.render_dataset_controls()

        # for debug only
        if DEBUG:
            st.session_state

    # main panel
    with st.container():
        # st_utils.update_state()
        # if st.session_state["ready_to_tango"]:
        st_utils.update_main_panel()


def get_menu_items() -> dict:
    """fetches the menu items for the top-right side of the app."""
    readme_path = Path(__file__).parents[0].joinpath("about.md").resolve()
    with open(readme_path, "r") as f:
        readme_data = f.read()
    bug_report_url = (
        "https://github.com/udupa-varun/hdf5_explorer/issues/new/choose"
    )

    menu_items = {
        "Report a Bug": bug_report_url,
        "About": readme_data,
        "Get help": None,
    }

    return menu_items


main()
