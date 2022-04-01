from this import d
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import norm
from typing import List

# st.header("Bayesian Inference")

intro_txt = """
In this section we will be comparing 2 different Bayesian Inference techniques
1. Variational Inference
2. Markov Chain Monte Carlo
"""
N_SAMPLES = 250

st.session_state["dist_1"] = None
st.session_state["dist_2"] = None

def generate_dist():
    # Generate a simple bimodal distribution
    col1, _, col2 = st.columns([8,1,8])
    with col1:

        m1 = st.slider("Distribution 1: Mean", -5.,5., -2.)
        sd1 = st.slider("Distribution 1: Standard Deviation",1.,5.)
    with col2:

        m2 = st.slider("Distribution 2: Mean",-5.,5., 2.)
        sd2 = st.slider("Distribution 2: Standard Deviation", 1., 5. )
    
    if st.button("Generate Distribution"):
        st.session_state["dist_1"] = norm(m1,sd1)
        st.session_state["dist_2"] = norm(m2,sd2)
        
        d1_samples = np.random.normal(m1, sd1, N_SAMPLES)
        d2_samples = np.random.normal(m2, sd2, N_SAMPLES)
        all_samples = np.concatenate([d1_samples, d2_samples])
        st.write(all_samples.shape)
        fig = ff.create_distplot([all_samples],["Posterior Distribution"],bin_size=0.4)
        st.plotly_chart(fig)    

def mcmc_section():
    st.subheader("Markov Chain Monte Carlo")
    st.markdown(mcmc_intro_txt)
    
    

mcmc_intro_txt = """
    This section shows MCMC sampling in action and how it can allows us to 
    sample complicated distributions from the posterior. For this application, 
    the [Metropolis Hasting Algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) is used to perform the sampling.

    $$
    A(x',x) = \min \\bigg( 1, \\frac{f(x')Q(x|x')}{f(x)Q(x'|x)} \\bigg) \\\\ \\\\
    \\footnotesize{\\text{Acceptance Criteria of MH Algorithm}}
    $$
    ---
    
    ## Process
    1. Set the parameters of the posterior distribution
    2. Initialize the starting point randomly
    3. Perform $n$ iterations of sampling
    4. Discard Burn-in period and obtain parameters for the sample
    
    """

def pg3():
    st.header("Bayesian Inference")
    st.subheader("Generating Distribution")
    generate_dist()
    st.subheader("Variational Inference")

def mixture_pdf(x:float) -> float: 
    return 0.5*st.session_state["dist_1"].pdf(x) + 0.5*st.session_state["dist_2"].pdf(x)

def mcmc(start_pos: float, samples: int = 1000,) -> List[float]:
    transitions = np.random.normal(size=1000)
    u_s = np.random.uniform(size=1000)
    start_pos = 0
    samples = [start_pos]
    for t,u in zip(transitions,u_s):
        next_pos = start_pos + t
        pdf_ratio = mixture_pdf(next_pos)/mixture_pdf(start_pos)
        if u < pdf_ratio: # If ratio > 1, guaranteed to enter this block
            samples.append(next_pos)
            start_pos = next_pos
        else:
            samples.append(start_pos)
        
