import pandas as pd
import streamlit as st


@st.cache(allow_output_mutation=True)
def get_data():
    return pd.read_csv('data\OnlineRetail.csv')

