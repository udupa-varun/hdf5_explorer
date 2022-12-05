import streamlit as st


def display_st_error(msg: str):
    """wraps around st.error() with a default icon.

    :param msg: error message
    :type msg: str
    """
    st.error(msg, icon="🚨")


def display_st_warning(msg: str):
    """wraps around st.warning() with a default icon.

    :param msg: warning message
    :type msg: str
    """
    st.warning(msg, icon="⚠️")


def display_st_info(msg: str):
    """wraps around st.info() with a default icon.

    :param msg: warning message
    :type msg: str
    """
    st.info(msg, icon="ℹ️")
