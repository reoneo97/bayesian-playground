import streamlit as st

intro_txt = """
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


def pg2():
    st.title("Markov Chain Monte Carlo")

    st.markdown(intro_txt)
    generate_posterior()


def generate_posterior():
    st.slider("Variance", min_value=0, max_value=100)
