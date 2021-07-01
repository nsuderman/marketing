import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from chart_studio import plotly as py
import plotly.graph_objs as go



def run(df):
    st.markdown('Before diving into the analysis and coding we need to understand what our key metrics are, '
             'and more specifically what is our __North Star__ metrics.')
    st.markdown('*__The North Star Metric__ is the single metric that best captures the core value that our product '
                'delivers to customers.*')
    st.write('')
    st.write('This metric depends on your company’s product, position, targets & more. Airbnb’s North Star Metric is '
             'nights booked whereas for Facebook, it is daily active users')


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.markdown('____')
    st.markdown('__Monthly Revenue__')
    st.write('This is what our data looks like:')
    st.write(df.head(25))

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceYearMonth'] = df['InvoiceDate'].map(lambda date: 100 * date.year + date.month)
    df['Revenue'] = df['UnitPrice'] * df['Quantity']
    revenue_df = df.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()

    st.write('Revenue by Month:')
    st.write(revenue_df)


    col1, col2 = st.beta_columns(2)
    with col1:
        plot_data = [
            go.Scatter(x=revenue_df.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
                       y=revenue_df.query("InvoiceYearMonth < 201112")['Revenue'],)
        ]

        plot_layout = go.Layout(
            xaxis={"type": "category"},
            title='Montly Revenue',
            title_x = 0.5
        )
        fig = go.Figure(data=plot_data, layout=plot_layout)

        st.plotly_chart(fig)
    with col2:
        revenue_df['MonthlyGrowth'] = revenue_df['Revenue'].pct_change()
        # visualization - line graph
        plot_data = [
            go.Scatter(
                x=revenue_df.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
                y=revenue_df.query("InvoiceYearMonth < 201112")['MonthlyGrowth'],
            )
        ]

        plot_layout = go.Layout(
            xaxis={"type": "category"},
            title='Montly Growth Rate',
            title_x = 0.5
        )

        fig = go.Figure(data=plot_data, layout=plot_layout)
        st.plotly_chart(fig)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.markdown('____')
    st.markdown('__United Kingdom Active User Count__')

    uk_df = df.query("Country=='United Kingdom'").reset_index(drop=True)
    uk_monthly_active = uk_df.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()

    st.write('Active Users:')
    st.write(uk_monthly_active)

    # plotting the output
    plot_data = [
        go.Bar(
            x=uk_monthly_active.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
            y=uk_monthly_active.query("InvoiceYearMonth < 201112")['CustomerID'],
        )
    ]

    plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Active Customers',
        title_x=0.5
    )

    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.markdown('____')
    st.markdown('__United Kingdom Order Count__')
    uk_monthly_orders = uk_df.groupby('InvoiceYearMonth')['Quantity'].sum().reset_index()

    st.write('Order Count:')
    st.write(uk_monthly_orders)

    # plotting the output
    plot_data = [
        go.Bar(
            x=uk_monthly_orders.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
            y=uk_monthly_orders.query("InvoiceYearMonth < 201112")['Quantity'],
        )
    ]

    plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Total # of Orders',
        title_x=0.5
    )

    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)


    st.markdown('____')
    st.markdown('__United Kingdom Average Revenue__')
    uk_ave_rev = uk_df.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()

    st.write('Average Revenue:')
    st.write(uk_ave_rev)

    # plotting the output
    plot_data = [
        go.Bar(
            x=uk_ave_rev.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
            y=uk_ave_rev.query("InvoiceYearMonth < 201112")['Revenue'],
        )
    ]

    plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Order Average',
        title_x=0.5
    )

    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.markdown('____')
    st.markdown('__United Kingdom New Customer Ratio__')
    tx_min_purchase = uk_df.groupby('CustomerID').InvoiceDate.min().reset_index()
    tx_min_purchase.columns = ['CustomerID', 'MinPurchaseDate']
    tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate']\
        .map(lambda date: 100 * date.year + date.month)

    # merge first purchase date column to our main dataframe (tx_uk)
    uk_df = pd.merge(uk_df, tx_min_purchase, on='CustomerID')

    uk_df['UserType'] = 'New'
    uk_df.loc[uk_df['InvoiceYearMonth'] > uk_df['MinPurchaseYearMonth'], 'UserType'] = 'Existing'

    st.write('Added in First Purchase Data and Flag for Existing Customer:')
    st.write(uk_df.head(25))

    st.write('Revenue by Customer Type:')
    uk_user_type_revenue = uk_df.groupby(['InvoiceYearMonth', 'UserType'])['Revenue'].sum().reset_index()
    st.write(uk_user_type_revenue)

    # filtering the dates and plot the result
    uk_user_type_revenue = uk_user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")
    plot_data = [
        go.Scatter(
            x=uk_user_type_revenue.query("UserType == 'Existing'")['InvoiceYearMonth'],
            y=uk_user_type_revenue.query("UserType == 'Existing'")['Revenue'],
            name='Existing'
        ),
        go.Scatter(
            x=uk_user_type_revenue.query("UserType == 'New'")['InvoiceYearMonth'],
            y=uk_user_type_revenue.query("UserType == 'New'")['Revenue'],
            name='New'
        )
    ]

    plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New vs Existing',
        title_x=0.5
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    st.write('Existing customers are showing a positive trend and tell us that our customer base is growing but '
             'new customers have a slight negative trend.')

    uk_user_ratio = uk_df.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique() / \
                    uk_df.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()
    uk_user_ratio = uk_user_ratio.reset_index()
    uk_user_ratio = uk_user_ratio.dropna()

    # print the dafaframe
    st.write('New Customer ratio:')
    st.write(uk_user_ratio)

    # plot the result

    plot_data = [
        go.Bar(
            x=uk_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['InvoiceYearMonth'],
            y=uk_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['CustomerID'],
        )
    ]

    plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New Customer Ratio',
        title_x=0.5
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.markdown('____')
    st.markdown('__United Kingdom Retention Rate__')
    st.write('')
    st.latex(r'''
        \textbf{Monthly Rention Rate} = Retained Customer From Prev Month  /  Active Customer Total
    ''')

    # identify which users are active by looking at their revenue per month
    uk_user_purchase = uk_df.groupby(['CustomerID', 'InvoiceYearMonth'])['Revenue'].sum().reset_index()

    # create retention matrix with crosstab
    uk_retention = pd.crosstab(uk_user_purchase['CustomerID'], uk_user_purchase['InvoiceYearMonth']).reset_index()
    st.write('Retained Customers From Last Month')
    st.write(uk_retention)

    # create an array of dictionary which keeps Retained & Total User count for each month
    months = uk_retention.columns[2:]
    retention_array = []
    for i in range(len(months) - 1):
        retention_data = {}
        selected_month = months[i + 1]
        prev_month = months[i]
        retention_data['InvoiceYearMonth'] = int(selected_month)
        retention_data['TotalUserCount'] = uk_retention[selected_month].sum()
        retention_data['RetainedUserCount'] = \
        uk_retention[(uk_retention[selected_month] > 0) & (uk_retention[prev_month] > 0)][selected_month].sum()
        retention_array.append(retention_data)

    uk_retention = pd.DataFrame(retention_array)
    uk_retention['RetentionRate'] = uk_retention['RetainedUserCount'] / uk_retention['TotalUserCount']
    st.write(uk_retention)

    # plot the retention rate graph
    plot_data = [
        go.Scatter(
            x=uk_retention.query("InvoiceYearMonth<201112")['InvoiceYearMonth'],
            y=uk_retention.query("InvoiceYearMonth<201112")['RetentionRate'],
            name="organic"
        )

    ]

    plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Retention Rate',
        title_x=0.5
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)








