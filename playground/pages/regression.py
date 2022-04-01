import streamlit as st
from ..data import regression_dataset
import plotly.express as px


intro_txt = """
    This section shows Bayesian Linear Regression and how it works! There are 
    2 main takeaways from this section:

    1. Show how Bayesian techniques return a posterior distribution instead of 
    point estimates
    2. Identify how different priors will lead to different results
    3. (?) Look at how different points will afe
    
    """


def pg1():

    st.title("Bayesian Playground")

    st.markdown(intro_txt)

    def display_data():
        X, y, coef = regression_dataset()
        go = px.scatter(x=X[:, 0], y=y)
        return go

    st.plotly_chart(display_data())
