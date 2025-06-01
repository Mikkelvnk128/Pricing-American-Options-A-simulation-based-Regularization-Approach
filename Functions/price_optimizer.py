from scipy.optimize import minimize_scalar

# Own import
from core_pricing_routine import american_price_inner 

def american_price(sim_params,opt_params,reg_params,model='Black-Scholes',heston_params=None,seed=None,out_of_sample=False):
    '''
    A small wrapper used for optimizing the p paramter in RSLM and DRLSM
    For simplicity the whole function goes through this function while LSM and DLSM just skips the step of traning p
    '''
    
    if reg_params['method'] in ('RLSM','DRLSM'):
        # reduced problem for p search
        reduced = sim_params.copy()
        reduced['num_paths']  = max(100,sim_params['num_paths']//4) # Smaller pathsize for optimizing 
        reduced['time_steps'] = max(3, sim_params['time_steps']//4) # Smaller pathsize for optimizing 

        def obj(pval): # Objective is to minmizie the negative of the price (Maximizing price)
            rp = reg_params.copy()
            rp['p'] = pval
            return -american_price_inner(reduced,opt_params,rp,model,heston_params,seed,False)[0]

        res = minimize_scalar(obj,bounds=(0.01, 5),method='bounded',options={'xatol':0.1}) # Scipy optimization
        reg_params['p'] = res.x

    return american_price_inner(sim_params,opt_params,reg_params,model,heston_params,seed,out_of_sample)