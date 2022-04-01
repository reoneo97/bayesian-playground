import streamlit as st
import plotly.express as px
from playground.data import regression_dataset
from playground.pages import pages


def update_page():
    pages[pg]()


with st.sidebar:

    st.header("Page Navigation")
    pg = st.radio("Page Selection",
                  options=["Bayesian Inference",
                        "Bayesian Linear Regression",
                        "Markov Chain Monte Carlo"],
                  # on_change=update_page
                  )


pages[pg]()
