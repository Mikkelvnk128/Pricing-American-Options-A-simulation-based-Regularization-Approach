from models import simulate_gbm, simulate_heston
from payoff import payoff

import numpy as np


def EOP(sim_params,opt_params,seed,model='Black-Scholes',heston_params=None):
    if seed is not None: np.random.seed(seed)
    
    if model == 'Black-Scholes':
        S_paths = simulate_gbm(sim_params)                  
    elif model == 'Heston':
        S_paths, _ = simulate_heston(sim_params,heston_params)
        
    S_T   = S_paths[:,-1] 
                   
    payoffs = payoff(S_T,opt_params["strike"],opt_params["style"])

    disc   = np.exp(-sim_params['r']*sim_params['T'])
    price  = disc*payoffs.mean()              
    return price

