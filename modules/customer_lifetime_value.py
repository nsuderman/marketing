import streamlit as st
import pandas as pd
from datetime import datetime, timedelta,date
from chart_studio import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


#order cluster method
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


#@st.cache(suppress_st_warning=True)
def analysis(df):
    st.write('We invest in customers (acquisition costs, offline ads, promotions, discounts & etc.) to '
             'generate revenue and be profitable. Naturally, these actions make some customers super valuable in '
             'terms of lifetime value but there are always some customers who pull down the profitability. '
             'We need to identify these behavior patterns, segment customers and act accordingly.')
    st.latex(r'''
                    \textbf{Lifetime Value} = Total Gross Revenue  -  Total Cost
                ''')

    st.markdown('__Lifetime Value Prediction__')

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
    tx_uk = df.query("Country=='United Kingdom'").reset_index(drop=True)

    # create 3m and 6m dataframes
    tx_3m = tx_uk[(tx_uk.InvoiceDate < date(2011, 6, 1)) & (tx_uk.InvoiceDate >= date(2011, 3, 1))].reset_index(
        drop=True)
    tx_6m = tx_uk[(tx_uk.InvoiceDate >= date(2011, 6, 1)) & (tx_uk.InvoiceDate < date(2011, 12, 1))].reset_index(
        drop=True)

    # create tx_user for assigning clustering
    tx_user = pd.DataFrame(tx_3m['CustomerID'].unique())
    tx_user.columns = ['CustomerID']

    # calculate recency score
    tx_max_purchase = tx_3m.groupby('CustomerID').InvoiceDate.max().reset_index()
    tx_max_purchase.columns = ['CustomerID', 'MaxPurchaseDate']
    tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
    tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID', 'Recency']], on='CustomerID')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Recency']])
    tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

    tx_user = order_cluster('RecencyCluster', 'Recency', tx_user, False)

    # calcuate frequency score
    tx_frequency = tx_3m.groupby('CustomerID').InvoiceDate.count().reset_index()
    tx_frequency.columns = ['CustomerID', 'Frequency']
    tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Frequency']])
    tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

    tx_user = order_cluster('FrequencyCluster', 'Frequency', tx_user, True)

    # calcuate revenue score
    tx_3m['Revenue'] = tx_3m['UnitPrice'] * tx_3m['Quantity']
    tx_revenue = tx_3m.groupby('CustomerID').Revenue.sum().reset_index()
    tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Revenue']])
    tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
    tx_user = order_cluster('RevenueCluster', 'Revenue', tx_user, True)

    # overall scoring
    tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
    tx_user['Segment'] = 'Low-Value'
    tx_user.loc[tx_user['OverallScore'] > 2, 'Segment'] = 'Mid-Value'
    tx_user.loc[tx_user['OverallScore'] > 4, 'Segment'] = 'High-Value'

    # calculate revenue and create a new dataframe for it
    tx_6m['Revenue'] = tx_6m['UnitPrice'] * tx_6m['Quantity']
    tx_user_6m = tx_6m.groupby('CustomerID')['Revenue'].sum().reset_index()
    tx_user_6m.columns = ['CustomerID', 'm6_Revenue']

    # plot LTV histogram
    plot_data = [
        go.Histogram(
            x=tx_user_6m.query('m6_Revenue < 10000')['m6_Revenue']
        )
    ]

    plot_layout = go.Layout(
        title='6m Revenue',
        title_x=0.5
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    tx_merge = pd.merge(tx_user, tx_user_6m, on='CustomerID', how='left')
    tx_merge = tx_merge.fillna(0)

    st.write(tx_merge)

    tx_graph = tx_merge.query("m6_Revenue < 30000")

    st.write('')
    st.write('Positive Correlation between High RFM and High LTV')
    plot_data = [
        go.Scatter(
            x=tx_graph.query("Segment == 'Low-Value'")['OverallScore'],
            y=tx_graph.query("Segment == 'Low-Value'")['m6_Revenue'],
            mode='markers',
            name='Low',
            marker=dict(size=7,
                        line=dict(width=1),
                        color='blue',
                        opacity=0.8
                        )
        ),
        go.Scatter(
            x=tx_graph.query("Segment == 'Mid-Value'")['OverallScore'],
            y=tx_graph.query("Segment == 'Mid-Value'")['m6_Revenue'],
            mode='markers',
            name='Mid',
            marker=dict(size=9,
                        line=dict(width=1),
                        color='green',
                        opacity=0.5
                        )
        ),
        go.Scatter(
            x=tx_graph.query("Segment == 'High-Value'")['OverallScore'],
            y=tx_graph.query("Segment == 'High-Value'")['m6_Revenue'],
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
        yaxis={'title': "6m LTV"},
        xaxis={'title': "RFM Score"},
        title='LTV',
        title_x=0.5
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

    st.markdown('____')

    st.write('We will use a XGBoost model to predict the LTV for our customers based on 3 segments.')
    st.write('Low LTV')
    st.write('Mid LTV')
    st.write('Hight LTV')

    # remove outliers
    tx_merge = tx_merge[tx_merge['m6_Revenue'] < tx_merge['m6_Revenue'].quantile(0.99)]

    # creating 3 clusters
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(tx_merge[['m6_Revenue']])
    tx_merge['LTVCluster'] = kmeans.predict(tx_merge[['m6_Revenue']])

    # order cluster number based on LTV
    tx_merge = order_cluster('LTVCluster', 'm6_Revenue', tx_merge, True)

    # creatinga new cluster dataframe
    tx_cluster = tx_merge.copy()

    # see details of the clusters
    st.write(tx_cluster.groupby('LTVCluster')['m6_Revenue'].describe())
    st.write('2 is the best with average 8.2k LTV whereas 0 is the worst with 396')

    # convert categorical columns to numerical
    tx_class = pd.get_dummies(tx_cluster)
    df1 = tx_class.pop('m6_Revenue')
    tx_class['m6_Revenue'] = df1

    # calculate and show correlations
    corr_matrix = tx_class.corr()
    corr_matrix['LTVCluster'].sort_values(ascending=False)
    st.write(corr_matrix)

    # View Correlation
    matrix = np.triu(tx_class.corr())
    fig = plt.figure()
    g = sns.heatmap(tx_class.corr(), annot=True, mask=matrix, cmap='coolwarm', annot_kws={"size": 4}
                    , fmt='.2f')
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=4, rotation=45)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=4, rotation=45)

    st.pyplot(fig)
    # plt.title(f'Correlation Matrix')

    # create X and y, X will be feature set and y is the label - LTV
    st.write(tx_class)
    X = tx_class.drop(['LTVCluster', 'm6_Revenue'], axis=1)
    y = tx_class['LTVCluster']
    return X, y


def run(df):
    X, y = analysis(df)

    # split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

    st.markdown("____")
    st.subheader("Using XGBoost Classifier to predict future LTV Segments for our Customers")
    # XGBoost Multiclassification Model


    max_depth = st.sidebar.slider('Max Depth',min_value=1,max_value=100,value=40)
    gamma = st.sidebar.slider('Gamma',min_value=0.0,max_value=1.0,value=0.7)
    min_child_weight = st.sidebar.slider('Min Child Weight',min_value=1,max_value=10,value=6)
    learning_rate = st.sidebar.slider('Learning Rate',min_value=.05,max_value=.3,value=.1)

    model = xgb.XGBClassifier(max_depth=max_depth, gamma=gamma,min_child_weight=min_child_weight,
                                      learning_rate=learning_rate, objective='multi:softprob', n_jobs=-1)\
        .fit(X_train, y_train)

    st.write('Accuracy of XGB classifier on training set: {:.2f}'
          .format(model.score(X_train, y_train)))
    st.write('Accuracy of XGB classifier on test set: {:.2f}'
          .format(model.score(X_test[X_train.columns], y_test)))

    y_pred = model.predict(X_test)
    st.table(classification_report(y_test, y_pred,output_dict=True))

    pickle_model = st.sidebar.button("Save Model")
    if pickle_model:
        pickle.dump(model, open("model.p", "wb"))
        st.markdown('____')
        st.header('Model Saved for Prediction Use')



