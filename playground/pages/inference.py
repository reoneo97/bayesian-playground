import streamlit as st
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import norm
from typing import List

import pyro 
import pyro.distributions as dist

# st.header("Bayesian Inference")

intro_txt = """
In this section we will be comparing 2 different Bayesian Inference techniques
1. Variational Inference
2. Markov Chain Monte Carlo
"""
N_SAMPLES = 250

st.session_state["dist_1"] = None
st.session_state["dist_2"] = None
st.session_state["mu_1"] = None
st.session_state["mu_2"] = None
st.session_state["sd_1"] = None
st.session_state["sd_2"] = None
st.session_state["MCMC_Samples"] = None
st.session_state["True_Samples"] = None

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
        st.session_state["True_Samples"] = all_samples
    
    if st.session_state["True_Samples"] is not None:
        fig = ff.create_distplot([st.session_state["True_Samples"]], 
                                 ["Posterior Distribution"],
                                 bin_size=0.4,show_rug=False)
        st.plotly_chart(fig)    

def mcmc_section():
    st.subheader("Markov Chain Monte Carlo")
    st.markdown(mcmc_intro_txt)
    start_pos = st.slider("MCMC Initial Position", -5., 5., 0.)
    # Ideally checking session states should become a decorator function if its used frequently
    if st.session_state["dist_1"]:
        if st.button("Run MCMC") :

            mcmc_samples = mcmc(start_pos)
            st.session_state["MCMC_Samples"] = mcmc_samples
        if st.session_state["MCMC_Samples"] is not None:
            sample_length = len(st.session_state["MCMC_Samples"])
            s_start, s_end = st.slider(
                "Exclude Burn-in Period", 0, sample_length-1,
                value=(sample_length//2, sample_length-1)
            )
            st.write(len(st.session_state["MCMC_Samples"]))
            st.write(s_start, s_end)
            mcmc_samples = st.session_state["MCMC_Samples"][s_start:s_end-1]
            fig = ff.create_distplot([mcmc_samples], ["Posterior Distribution"],
                                     bin_size=0.4, show_rug=False)
            st.plotly_chart(fig)


    
    

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


def mixture_pdf(x:float) -> float: 
    """Helper function to generate the pdf from the mixture distribution. Assumes
    that the mixture weights are both 0.5

    Args:
        x (float): Position of the evaluated point

    Returns:
        float: Density of the point with mixture distribution
    """
    return 0.5*st.session_state["dist_1"].pdf(x) + 0.5*st.session_state["dist_2"].pdf(x)

def mcmc(start_pos: float, samples: int = 5000,) -> List[float]:
    transitions = np.random.normal(size=samples)
    u_s = np.random.uniform(size=samples)
    samples = []
    for t,u in zip(transitions,u_s):
        next_pos = start_pos + t
        pdf_ratio = mixture_pdf(next_pos)/mixture_pdf(start_pos)
        if u < pdf_ratio: # If ratio > 1, guaranteed to enter this block
            samples.append(next_pos)
            start_pos = next_pos
        else:
            samples.append(start_pos)
    return samples
    
        

def pg3():
    st.header("Bayesian Inference")
    st.subheader("Generating Distribution")
    generate_dist()
    st.subheader("Variational Inference")
    mcmc_section()

def mean_field1():
    """
    Initial Mean Field Inference by only using 1 Gaussian to approximate the 
    posterior. Implemented using Pyro. 
    """
    def model(data):
        # Ground Truth
        for i in pyro.plate("data_loop",len(data)):

            ber = pyro.sample(dist.Bernoulli(0.5), obs=data[i])


def mean_field2():
    """
    More advanced model to use 2 gaussians and a beta distribution to 
    approximate the posterior
    """
    pass