import streamlit as st
import numpy as np
import pandas as pd

st.title("FIFA 2018 Player Skills")
st.markdown(
"""
This is a project that uses ML to determine player pay based on their differnt skill levels
""")

data_location = ('FIFA2018.csv')

def load_data(nrows):
    data = pd.read_csv(data_location, encoding = 'latin1', nrows=nrows)
    return data

data_points = st.slider('data points', 0, 100, 50)

data = load_data(data_points)


st.write("Here's our first attempt at using data to create a table:")
st.dataframe(data)

