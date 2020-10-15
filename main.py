#importing app dependencies 
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data_location = ('cleaned_data.csv')

def main():
    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Data', 'Kmeans'])

    if page == 'Homepage':
        #some formatting for the webapp
        st.title("FIFA 2018 Player Skills")
        img = "media/FIFA.jpg"
        st.image(img, width = 400)
        st.markdown(
        """
        This is a project that uses ML to determine player pay based on their differnt skill levels
        """)

    if page == "Data":

        #storing the location of our cleaned data 

        #lets user pick how many rows they want to see in the app
        def load_data(nrows):
            data = pd.read_csv(data_location, encoding = 'latin1', nrows=nrows)
            return data

        #allows user to toggle visibility of the data 
        if st.checkbox('Show raw data'):
            with st.beta_container():
                data_points = st.slider('data points', 0, 100, 50)
                data = load_data(data_points)
                st.subheader('Raw data')
                st.dataframe(data)
    if page == "Kmeans":
        df = pd.read_csv(data_location)

        #only considering FIFA stats for clustering
        features_to_cluster = df.loc[:,'Crossing':'GKReflexes'].columns

        cluster_df = df.loc[:,features_to_cluster]
        cluster_df['Name'] = df['Name']

        #generating a list of features to append to a user selection drop down menu
        feature_list = list(cluster_df.columns) 
        feature_list.remove('Name')

        #sidebar multiselection to allow user to pick which features to use for clustering
        chosen_features = st.multiselect('Please choose two features to compare', feature_list)
        if len(chosen_features) < 2:
            st.warning('Please pick two features to proceed.')
            st.stop()

        chosen_feature1 = chosen_features[0]
        chosen_feature2 = chosen_features[1]

        #df is filtered to two user inpits plus name column
        df_chosen = cluster_df[[chosen_feature1, chosen_feature2, 'Name']]

        #next user selects the k value for clustering
        k = st.slider('Pick a K value', 2, 8, 5)
        kmeans = KMeans(n_clusters=k, random_state=0)
        y_pred = kmeans.fit_predict(df_chosen.drop('Name', axis=1))

        #code adapted from Aurelien Geron's book 'Hands on Machine Learning with Scikit-Learn, Keras, and TensorFlow'
        #below is for Matplot

        with st.beta_expander("Matplot"):
            def plot_data(df_chosen):
                plt.plot(df_chosen.loc[:,chosen_feature1], df_chosen.loc[:,chosen_feature2], 'k.', markersize=2)

            def plot_centroids(centroids, circle_color='w', cross_color='k'):
                plt.scatter(centroids[:, 0], centroids[:, 1],
                            marker='o', s=30, linewidths=8,
                            color=circle_color, zorder=10, alpha=0.9)
                plt.scatter(centroids[:, 0], centroids[:, 1],
                            marker='x', s=50, linewidths=50,
                            color=cross_color, zorder=11, alpha=1)

            def plot_decision_boundaries(clusterer, df_chosen, resolution=1000, show_centroids=True,
                                        show_xlabels=True, show_ylabels=True):
                mins = df_chosen.drop('Name', axis=1).min(axis=0) - 0.1
                maxs = df_chosen.drop('Name', axis=1).max(axis=0) + 0.1
                xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                                    np.linspace(mins[1], maxs[1], resolution))
                Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                            cmap="Pastel2")
                plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                            linewidths=1, colors='k')
                plot_data(df_chosen)
                if show_centroids:
                    plot_centroids(clusterer.cluster_centers_)

                if show_xlabels:
                    plt.xlabel(f"{chosen_feature1}", fontsize=14)
                else:
                    plt.tick_params(labelbottom=False)
                if show_ylabels:
                    plt.ylabel(f"{chosen_feature2}", fontsize=14, rotation=90)
                else:
                    plt.tick_params(labelleft=False)

            plt.figure(figsize=(8, 4))
            plot = plot_decision_boundaries(kmeans, df_chosen)

            #needed to remove PyplotGlobalUseWarning
            st.set_option('deprecation.showPyplotGlobalUse', False)


            st.pyplot(plot)

        #below is for Plot.ly

        with st.beta_expander("Plotly"):

            def plot_decision_boundaries_plotly(clusterer, df_chosen, player=None):
                x_min, x_max = df_chosen.loc[:, chosen_feature1].min() - 1, df_chosen.loc[:, chosen_feature1].max() + 1
                y_min, y_max = df_chosen.loc[:, chosen_feature2].min() - 1, df_chosen.loc[:, chosen_feature2].max() + 1
                #maxs = df_chosen.max(axis=0) + 0.1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, .02)
                                , np.arange(y_min, y_max, .02))
                y_ = np.arange(y_min, y_max, 0.02)
                Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                trace1 = go.Heatmap(x=xx[0], y=y_, z=Z,
                            colorscale='Viridis',
                            showscale=True)

                trace2 = go.Scatter(x=df_chosen.loc[:,chosen_feature1], y=df_chosen.loc[:,chosen_feature2], 
                                mode='markers',
                                text=df['Name'],
                                marker=dict(size=10,
                                            color=df[chosen_feature1], 
                                            colorscale='Viridis',
                                            line=dict(color='black', width=1))
                                            )

                if player:
                    trace3 = go.Scatter(x=df_chosen.loc[df_chosen['Name'] == player, chosen_feature1], y=df_chosen.loc[df_chosen['Name'] == player, chosen_feature2],
                            mode='markers',
                            marker=dict(size=20,
                                        color='red',
                                        line=dict(color='black', width=2))
                    )
                    data = [trace1, trace2, trace3]
                else:
                    data= [trace1, trace2]

                layout= go.Layout(
                autosize= True,
                title= 'K-Means',
                hovermode= 'closest',
                showlegend= False)

                #data = [trace1, trace2]
                #fig = go.Figure(data=data, layout=layout)

            player = st.text_input("which player are you interested in?", "L. Massi")
                
            plot2 = plot_decision_boundaries_plotly(kmeans, df_chosen, player=player)

            st.plotly_chart(plot2)

if __name__ == '__main__':
    main()

