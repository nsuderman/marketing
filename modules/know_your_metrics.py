import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
from datetime import datetime, timedelta


def run(df):
    st.markdown('Before diving into the analysis and coding we need to understand what our key metrics are, '
             'and more specifically what is our __North Star__ metric.')
    st.write('Before diving into the analysis and coding we need to understand what our key metrics are, '
             'and more specifically what is our **North Star** metric.')