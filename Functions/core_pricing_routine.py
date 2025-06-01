import numpy as np
from scipy.stats import gmean

# Own imports 
from models import simulate_gbm, simulate_heston
from payoff import payoff
from regression import fit_regression


def american_price_inner(sim_params, opt_params, reg_params, 
                             model='Black-Scholes', heston_params=None,
                                 seed=None, out_of_sample=False):
    
    r          = sim_params['r']
    num_paths  = sim_params['num_paths']
    time_steps = sim_params['time_steps']
    strike         = opt_params['strike']
    style          = opt_params.get('style', 'put') # Fall back to put
    num_basis      = reg_params['num_basis']
    method         = reg_params['method']
    include_payoff = reg_params.get('include_payoff',False) # Fall back to no payoff in regression
    p              = reg_params.get('p', 4) # Fall back to p of 4
    dt             = sim_params['T']/time_steps   

    # seed
    if seed is not None: np.random.seed(seed)

    # activation functions
    eps = 1e-8
    xp = lambda x: np.maximum(x,eps) # Safe divide
    activation = lambda x: xp(x)*(1-xp(x)**p)/(1-xp(x))
    dactivation = lambda x: (1-(p+1)*xp(x)**p+p*xp(x)**(p+1))/(1-xp(x))**2

    discount = np.exp(-r*dt)

    # simulate training paths
    if model == 'Heston':
        stock_paths,v_paths = simulate_heston(sim_params,heston_params)
    else: # GBM (Black-Scholes)
        stock_paths = simulate_gbm(sim_params)
        v_paths      = None

    # Amount of underlyings (d)
    d = stock_paths.shape[2] if stock_paths.ndim == 3 else 1

    # terminal payoffs and cashflow matrix
    if stock_paths.ndim > 2:
        S_term = stock_paths[:,-1,:]
    else:
        S_term = stock_paths[:,-1]

    cashflow_mat = payoff(S_term,strike,style)
    cashflows    = cashflow_mat.copy()

    # continuation indicator matrix
    continuation_values = np.zeros((num_paths,time_steps+1))
    continuation_values[:,-1] = cashflow_mat>0

    # prepare for RLSM/DRLSM
    if method in ('RLSM', 'DRLSM'):
        A = np.random.lognormal(mean=0,sigma=0.01,size=(d,num_basis))
        reg_params['A'] = A
        reg_params['act'] = activation
    else:
        A = None

    # prepare for DLSM/DRLSM
    if method in ('DLSM', 'DRLSM'):
        if stock_paths.ndim > 2:
            STauStar = np.where(continuation_values[:,-1].reshape(-1,1)==1,stock_paths[:,-1,:],0)
        else:
            STauStar = np.where(continuation_values[:,-1].reshape(-1,1)==1,stock_paths[:,-1].reshape(-1,1),0)
    else:
        STauStar = None

    regression_coeffs = {}

    # backward induction
    for t in range(time_steps-1,0,-1):
        
        # compute in-the-money prices (Find the aggregated style to find itm paths)
        if stock_paths.ndim > 2:
            price_agg = {'geocall':gmean,'geoput':gmean,'minput':np.min,'maxcall':np.max}[style](stock_paths[:,t,:],axis=1)
        else:
            price_agg = stock_paths[:,t]

        itm = np.where((price_agg<strike) if 'put' in style else (price_agg>strike))[0]
        
        if itm.size == 0: # If no itm then skip
            continue

        cf = cashflows[itm]*discount

        if method in ('DLSM', 'DRLSM'):
            STauStar[itm] *= discount

        # Price dynamics at itm
        if stock_paths.ndim > 2:
            S_t = stock_paths[itm,t,:]
        else:
            S_t = stock_paths[itm,t].reshape(-1, 1)

        V_t = None
        if model == 'Heston':
            if stock_paths.ndim > 2:
                V_t = v_paths[itm,t,:]
            else:
                V_t = v_paths[itm,t].reshape(-1, 1)

        # regression to get estimated beta (beta_hat) and basis function (phi)
        beta_hat, phi = fit_regression(
            reg_params,S_t,cf,strike,discount,style, # Standard inputs
            model=model, v_heston=V_t,# Model inputs 
            STauStar=STauStar[itm] if STauStar is not None else None, # For delta methods
            A=A, activation=activation, dactivation=dactivation # For RLSM methods
        )

        regression_coeffs[t] = beta_hat # Store coefficients for out-of-sample

        continuation = phi@beta_hat
        continuation_values[itm,t] = continuation

        # immediate payoff for all paths
        if stock_paths.ndim > 2:
            immediate = payoff(stock_paths[:,t,:],strike,style)
        else:
            immediate = payoff(stock_paths[:,t],strike,style)

        # exercise decision
        exercise_now = immediate>continuation_values[:,t] # Continutation descision 
        cashflows    = np.where(exercise_now,immediate,cashflows*discount)
        continuation_values[:,t] = exercise_now

        # zero out future for exercised
        exercised_idx = np.where(continuation_values[:,t]==1)[0]
        if exercised_idx.size>0:
            continuation_values[exercised_idx,t+1:]=0

        # update STauStar for delta methods
        if method in ('DLSM', 'DRLSM'):
            if stock_paths.ndim > 2:
                current_prices = stock_paths[:,t,:]
            else:
                current_prices = stock_paths[:,t].reshape(-1, 1)
                
                STauStar = np.where(continuation_values[:,t:t+1]==1,current_prices,STauStar)

    # discount in-sample cashflows
    in_sample_price = cashflows.mean()*discount

    # Now do out of sample (if requested) for stored coeffecients 
    if out_of_sample:
        if seed is not None:
            np.random.seed(seed+1000)

        if model == 'Heston':
            pricing_paths,v_pricing = simulate_heston(sim_params,heston_params)
        else:
            pricing_paths = simulate_gbm(sim_params)
            v_pricing     = None

        # terminal payoff & continuation indicator
        if pricing_paths.ndim > 2:
            p_terminal = pricing_paths[:,-1,:]
        else:
            p_terminal = pricing_paths[:,-1]

        pricing_cashflows = payoff(p_terminal,strike,style)
        continuation_values = np.zeros((num_paths, time_steps+1))
        continuation_values[:, -1] = pricing_cashflows>0

        # backward pass
        for t in range(time_steps-1,0,-1):
            if t not in regression_coeffs:
                continue # if no regression coefficients for current t (no itm in training) then skip step t

            if pricing_paths.ndim > 2:
                agg_t = {'geocall':gmean,'geoput':gmean,'minput':np.min,'maxcall':np.max}[style](pricing_paths[:,t,:],axis=1)
            else:
                agg_t = pricing_paths[:,t]

            itm = np.where((agg_t<strike) if 'put' in style else (agg_t>strike))[0]
            if itm.size == 0:
                continue

            if pricing_paths.ndim > 2:
                S_price = pricing_paths[itm,t,:]
            else:
                S_price = pricing_paths[itm,t].reshape(-1, 1)

            # build phi using stored A, activation
            if method in ('RLSM', 'DRLSM'): # Phi has same structure for both methods
            
                phi = np.hstack([activation(S_price@A),np.ones((S_price.shape[0],1))])
                
                if include_payoff:
                    immediate_payoff = payoff(S_price,strike,style).reshape(-1,1)
                    phi = np.hstack([phi, immediate_payoff])
                    
                if model == 'Heston':
                    if pricing_paths.ndim > 2:
                        V_phi = v_pricing[itm,t,:]
                    else:
                        V_phi = v_pricing[itm,t].reshape(-1,1)
                    phi = np.hstack([phi,activation(V_phi@A)])
                    
            else: # phi for LSM and DLSM
            
                phi = np.hstack([S_price**i for i in range(num_basis+1)])
                
                if include_payoff:
                    immediate_payoff = payoff(S_price,strike,style).reshape(-1,1)
                    phi = np.hstack([phi,immediate_payoff])
                    
                if model == 'Heston':
                    if pricing_paths.ndim>2:
                        Vp = v_pricing[itm,t,:]
                    else:
                        Vp = v_pricing[itm,t].reshape(-1,1)
                    vphi = np.hstack([Vp**i for i in range(1,num_basis+1)])
                    phi  = np.hstack([phi,vphi])

            beta_hat = regression_coeffs[t]
            continuation = phi@beta_hat
            continuation_values[itm,t] = continuation

            if pricing_paths.ndim>2:
                immediate = payoff(pricing_paths[:,t,:],strike,style)
            else:
                immediate = payoff(pricing_paths[:,t],strike,style)

            exercise_now = immediate>continuation_values[:,t]
            pricing_cashflows = np.where(exercise_now,immediate,pricing_cashflows*discount)
            continuation_values[:,t] = exercise_now

            exercised_idx = np.where(continuation_values[:,t]==1)[0]
            if exercised_idx.size>0:
                continuation_values[exercised_idx,t+1:] = 0

        out_of_sample_price = pricing_cashflows.mean()*discount
        return in_sample_price, out_of_sample_price, regression_coeffs

    return in_sample_price, regression_coeffs
