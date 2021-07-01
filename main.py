from data import data
import streamlit as st


if __name__ == '__main__':
    st.set_page_config(page_title="Marketing", layout='wide', initial_sidebar_state='auto')

    data = data.get_data()

    st.sidebar.header('Navigation')
    st.sidebar.write('') # Line break

    side_menu_selectbox = st.sidebar.selectbox('Menu',('Home',
                                                       'Know Your Metrics',
                                                       'Customer Segmentation',
                                                       'Customer Lifetime Value Prediction',
                                                       'Churn Prediction',
                                                       'Predicting Next Purchase Day',
                                                       'Predicting Sales',
                                                       'Market Response Models',
                                                       'Uplift Modeling',
                                                       'A/B Testing Design and Execution')
                                               )

    if side_menu_selectbox == 'Home':
        st.header("Project Overview")
        st.write('')
        st.write('This project was put together to demonstrate an agile approach to data science and analytics.')
        st.write('')
        st.write('In this data science web application you can navigate through the various sections')
        st.write('**Data Analysis:** In this section we will review our sales data and a few key metrics. ')
        st.write('')

        st.write('**Customer Segmentation:** In this section we will...')
        st.write('')

        st.write('**Customer Lifetime Value Prediction:** In this section we will...')
        st.write('')


    elif side_menu_selectbox == 'Know Your Metrics':
        st.header('Know Your Metrics')
        st.write('')

    elif side_menu_selectbox == 'Customer Segmentation':
        st.header('Customer Segmentation')
        st.write('')

    elif side_menu_selectbox == 'Customer Lifetime Value Prediction':
        st.header('Customer Lifetime Value Prediction')
        st.write('')

    elif side_menu_selectbox == 'Churn Prediction':
        st.header('Churn Prediction')
        st.write('')

    elif side_menu_selectbox == 'Predicting Next Purchase Day':
        st.header('Predicting Next Purchase Day')
        st.write('')

    elif side_menu_selectbox == 'Predicting Sales':
        st.header('Predicting Sales')
        st.write('')

    elif side_menu_selectbox == 'Market Response Models':
        st.header('Market Response Models')
        st.write('')

    elif side_menu_selectbox == 'Uplift Modeling':
        st.header('Uplift Modeling')
        st.write('')

    elif side_menu_selectbox == 'A/B Testing Design and Execution':
        st.header('A/B Testing Design and Execution')
        st.write('')


