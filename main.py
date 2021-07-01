from data import data
import streamlit as st
from modules import know_your_metrics as kym
from modules import segmentation as seg
from modules import customer_lifetime_value as clv
from streamlit.server.server import Server
from streamlit.report_thread import get_report_ctx
import os
import warnings
warnings.filterwarnings("ignore")

def watchers():
    # get report context
    ctx = get_report_ctx()
    # get session id
    session_id = ctx.session_id

    # get session
    server = Server.get_current()
    session_info = server._session_info_by_id.get(session_id)
    session = session_info.session

    # register watcher
    # session._local_sources_watcher._register_watcher(
    #     os.path.join(os.path.dirname(__file__), 'modules\know_your_metrics.py'),'dummy:modules\know_your_metrics.py')
    # session._local_sources_watcher._register_watcher(
    #     os.path.join(os.path.dirname(__file__), 'modules\segmentation.py'), 'dummy:modules\segmentation.py')
    # session._local_sources_watcher._register_watcher(
    #     os.path.join(os.path.dirname(__file__), 'modules\customer_lifetime_value.py'), 'dummy:modules\customer_lifetime_value.py')
    #

if __name__ == '__main__':
    st.set_page_config(page_title="Marketing", layout='wide', initial_sidebar_state='auto')
    watchers()

    # st.markdown('Bold: __bold__')
    # st.markdown('Italics: _italics_')
    # st.markdown('Bold & Italics: __*bold & italics*__')
    # st.markdown('Weblink: [Google](https://www.google.com/)')
    # st.subheader('A Section')
    # st.markdown('____')

    data = data.get_data()

    st.sidebar.header('Navigation')
    st.sidebar.write('') # Line break

    side_menu_selectbox = st.sidebar.selectbox('Menu',('Home',
                                                       'Know Your Metrics',
                                                       'Customer Segmentation',
                                                       'Customer Lifetime Value Prediction',
                                                       )
                                               )

    if side_menu_selectbox == 'Home':
        st.header("Project Overview")
        st.write('')
        st.write('This project was put together to demonstrate an agile approach to data science and analytics.')
        st.write('')
        st.write('In this data science web application you can navigate through the various sections')
        st.write('**Know Your Metrics:** In this section we will review our sales data and a few key metrics, sucah as '
                 'Monthly Revenue, Monthly Order Counts, New vs Existing Customers, and Rention Rate')
        st.write('')

        st.write('**Customer Segmentation:** In this section we will segment the customers base don the RFM model to '
                 'help better match communication and marketing to each customer group.')
        st.markdown('__RFM__ = Recency - Frequency - Monetary Value')
        st.write('')

        st.write('**Customer Lifetime Value Prediction:** In this section we will review the Customer Lifetime Value. '
                 'This metrics helps to segment customers and identify patterns.')
        st.write('')


    elif side_menu_selectbox == 'Know Your Metrics':
        st.header('Know Your Metrics')
        st.write('')

        kym.run(data)

    elif side_menu_selectbox == 'Customer Segmentation':
        st.header('Customer Segmentation')
        st.write('')

        seg.run(data)

    elif side_menu_selectbox == 'Customer Lifetime Value Prediction':
        st.header('Customer Lifetime Value Prediction')
        st.write('')

        clv.run(data)
