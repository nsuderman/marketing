import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from chart_studio import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans


#function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final



def run(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    tx_uk = df.query("Country=='United Kingdom'").reset_index(drop=True)

    st.markdown('__Recency__')
    st.write('To calculate recency, we need to find out most recent purchase date of each customer and see how many '
             'days they are inactive for. After having no. of inactive days for each customer, we will apply K-means* '
             'clustering to assign customers a recency score.')

    # create a generic user dataframe to keep CustomerID and new segmentation scores
    tx_user = pd.DataFrame(tx_uk['CustomerID'].unique())
    tx_user.columns = ['CustomerID']

    # get the max purchase date for each customer and create a dataframe with it
    tx_max_purchase = tx_uk.groupby('CustomerID').InvoiceDate.max().reset_index()
    tx_max_purchase.columns = ['CustomerID', 'MaxPurchaseDate']

    # we take our observation point as the max invoice date in our dataset
    tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

    # merge this dataframe to our new user dataframe
    tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID', 'Recency']], on='CustomerID')

    st.write('Recency by Customer:')
    st.write(tx_user)

    plot_data = [
        go.Histogram(
            x=tx_user['Recency']
        )
    ]

    plot_layout = go.Layout(
        title='Recency'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    st.write("Apply K-Means clustering to assign a recency score. The below chart shows how many clusters  are needed.")
    sse = {}
    tx_recency = tx_user[['Recency']]
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
        tx_recency["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_

    plot_data = [
        go.Scatter(
            x=list(sse.keys()),
            y=list(sse.values()),
        )
    ]

    plot_layout = go.Layout(
        xaxis={"type": "category"},
        xaxis_title="Number of Clusters",
        title='Clustering Inertia Graph',
        title_x=0.5
    )

    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    st.write('3 to 4 clusters appears to be the optimal grouping')
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Recency']])
    tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

    tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

    st.write('Recency Statistics:')
    st.write(tx_user.groupby('RecencyCluster')['Recency'].describe())
    st.write('Group 3 covers our most recent customers and group 0 is our most inactive customers')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.markdown('____')
    st.markdown('__Frequency__')
    st.write('To create frequency clusters, we need to find total number orders for each customer. I first calculate '
             'this and see how frequency look like in our customer database.')


    # get order counts for each user and create a dataframe with it
    tx_frequency = tx_uk.groupby('CustomerID').InvoiceDate.count().reset_index()
    tx_frequency.columns = ['CustomerID', 'Frequency']

    # add this data to our main dataframe
    tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')

    # plot the histogram
    plot_data = [
        go.Histogram(
            x=tx_user.query('Frequency < 1000')['Frequency']
        )
    ]

    plot_layout = go.Layout(
        title='Frequency',
        title_x=0.5
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)


    # k-means
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Frequency']])
    tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

    # order the frequency cluster
    tx_user = order_cluster('FrequencyCluster', 'Frequency', tx_user, True)

    # see details of each cluster
    st.write(tx_user)
    st.write(tx_user.groupby('FrequencyCluster')['Frequency'].describe())
    st.write('Similarly to the Recency, a higher frequency number indicates a better customer')


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.markdown('____')
    st.markdown('__Revenue__')
    st.write('Let’s see how our customer database looks like when we cluster them based on revenue.')

    # calculate revenue for each customer
    tx_uk['Revenue'] = tx_uk['UnitPrice'] * tx_uk['Quantity']
    tx_revenue = tx_uk.groupby('CustomerID').Revenue.sum().reset_index()

    # merge it with our main dataframe
    tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')

    # plot the histogram
    plot_data = [
        go.Histogram(
            x=tx_user.query('Revenue < 10000')['Revenue']
        )
    ]

    plot_layout = go.Layout(
        title='Monetary Value',
        title_x=0.5
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)


    # apply clustering
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Revenue']])
    tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])

    # order the cluster numbers
    tx_user = order_cluster('RevenueCluster', 'Revenue', tx_user, True)

    # show details of the dataframe
    st.write(tx_user.groupby('RevenueCluster')['Revenue'].describe())


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.markdown('____')
    st.markdown('__Overall Score__')
    st.write('We have scores (cluster numbers) for recency, frequency & revenue. Let’s create an '
             'overall score out of them: ')
    st.latex(r'''
            \textbf{Over All Score} = RecencyCluster  +  FrequencyCluster + RevenueCluster
        ''')
    st.write('')
    st.write(' - 0 to 2: Low Value Customer')
    st.write(' - 3 to 4: Mid Value Customer')
    st.write(' - 5+: High Value Customer')

    # calculate overall score and use mean() to see details
    tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
    st.write(tx_user.groupby('OverallScore')['Recency', 'Frequency', 'Revenue'].mean())

    tx_user['Segment'] = 'Low-Value'
    tx_user.loc[tx_user['OverallScore'] > 2, 'Segment'] = 'Mid-Value'
    tx_user.loc[tx_user['OverallScore'] > 4, 'Segment'] = 'High-Value'

    tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")

    plot_data = [
        go.Scatter(
            x=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
            y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
            mode='markers',
            name='Low',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='blue',
                        opacity=0.8
                        )
        ),
        go.Scatter(
            x=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
            y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
            mode='markers',
            name='Mid',
            marker=dict(size=9,
                        line=dict(width=1),
                        color='green',
                        opacity=0.5
                        )
        ),
        go.Scatter(
            x=tx_graph.query("Segment == 'High-Value'")['Frequency'],
            y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
            mode='markers',
            name='High',
            marker=dict(size=11,
                        line=dict(width=1),
                        color='red',
                        opacity=0.9
                        )
        ),
    ]

    plot_layout = go.Layout(
        yaxis={'title': "Revenue"},
        xaxis={'title': "Frequency"},
        title='Segments',
        showlegend=True,
        title_x=0.5
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)


    plot_data = [
        go.Scatter(
            x=tx_graph.query("Segment == 'Low-Value'")['Recency'],
            y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
            mode='markers',
            name='Low',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='blue',
                        opacity=0.8
                        )
        ),
        go.Scatter(
            x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],
            y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
            mode='markers',
            name='Mid',
            marker=dict(size=9,
                        line=dict(width=1),
                        color='green',
                        opacity=0.5
                        )
        ),
        go.Scatter(
            x=tx_graph.query("Segment == 'High-Value'")['Recency'],
            y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
            mode='markers',
            name='High',
            marker=dict(size=11,
                        line=dict(width=1),
                        color='red',
                        opacity=0.9
                        )
        ),
    ]

    plot_layout = go.Layout(
        yaxis={'title': "Revenue"},
        xaxis={'title': "Recency"},
        title='Segments',
        title_x=0.5
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)


    plot_data = [
        go.Scatter(
            x=tx_graph.query("Segment == 'Low-Value'")['Recency'],
            y=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
            mode='markers',
            name='Low',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='blue',
                        opacity=0.8
                        )
        ),
        go.Scatter(
            x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],
            y=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
            mode='markers',
            name='Mid',
            marker=dict(size=9,
                        line=dict(width=1),
                        color='green',
                        opacity=0.5
                        )
        ),
        go.Scatter(
            x=tx_graph.query("Segment == 'High-Value'")['Recency'],
            y=tx_graph.query("Segment == 'High-Value'")['Frequency'],
            mode='markers',
            name='High',
            marker=dict(size=11,
                        line=dict(width=1),
                        color='red',
                        opacity=0.9
                        )
        ),
    ]

    plot_layout = go.Layout(
        yaxis={'title': "Frequency"},
        xaxis={'title': "Recency"},
        title='Segments',
        title_x=0.5
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    st.write('High Value: Improve Retention')
    st.write('Mid Value: Improve Retention + Increase Frequency')
    st.write('Low Value: Increase Frequency')
