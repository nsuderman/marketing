import pandas as pd
import streamlit as st


@st.cache
def get_data():
    return pd.read_csv('data\OnlineRetail.csv')

