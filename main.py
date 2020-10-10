import streamlit as st
import numpy as np
import pandas as pd

st.title("FIFA 2018 Player Skills")

img = "media/FIFA.jpg"

st.image(img, width = 400)

st.markdown(
"""
This is a project that uses ML to determine player pay based on their differnt skill levels
""")

data_location = ('FIFA2018.csv')

def load_data(nrows):
    data = pd.read_csv(data_location, encoding = 'latin1', nrows=nrows)
    return data

if st.checkbox('Show raw data'):
    with st.beta_container():
        data_points = st.slider('data points', 0, 100, 50)
        data = load_data(data_points)
        st.subheader('Raw data')
        st.dataframe(data)


MLoption = st.sidebar.selectbox(
'What method of Unsupervised Learning do you want to use?',
('K-Means', 'Gaussian Mixture'))

country_select = st.sidebar.text_input(
    'Which country are you interested in?',
    value = "Spain"
)

club_select = st.sidebar.text_input(
    'Which club are you interested in?',
    value = "Real Madrid"
)
