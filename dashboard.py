
import streamlit as st
import pandas as pd
import numpy as np
# import plotly.figure_factory as ff
import matplotlib.pyplot as plt

# ------------------------------------------------
# conda activate datamining
# streamlit run dashboard.py
st.title('Food Demand Forecasting 1')
# ------------------------------------------------

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv('train.csv', nrows=nrows)
#     return data

# Feature Selection
menu = ['Assignment 1', 'Assignment 2']
st.sidebar.subheader('Main Menu')
page = st.sidebar.selectbox("Select Page Menu", menu)
st.sidebar.subheader('Group Member')
st.sidebar.text('Izzah\t1171101738 \nGlenn\t1171101736 \nNiroshaan\t1171101816 \nSyazwan1171101803')

if page == 'Assignment 1':
    st.title('Assignment 1 - Dashboard')

if page == 'Assignment 2':
    st.title('Assignment 2 - Dashboard')