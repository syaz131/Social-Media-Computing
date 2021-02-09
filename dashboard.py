
import streamlit as st
import pandas as pd
import numpy as np
# import plotly.figure_factory as ff
import matplotlib.pyplot as plt

# ------------------------------------------------
# conda activate datamining
# streamlit run "Assignment 1 - viz.py"
st.title('Food Demand Forecasting 1')
# ------------------------------------------------

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv('train.csv', nrows=nrows)
#     return data

# Feature Selection
menu = ['Assignment 1', 'Assignment 2']
st.sidebar.subheader('Main Menu')
page = st.sidebar.selectbox("Select Menu Page", menu)

if page == 'Assignment 1':
    st.title('Assignment 1 - Dashboard')

if page == 'Assignment 2':
    st.title('Assignment 2 - Dashboard')