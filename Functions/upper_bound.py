import numpy as np
from joblib import Parallel, delayed
from payoff import payoff


def dual_upper_bound(sim_params,opt_params,beta_dict,reg_params,model='Black-Scholes',heston_params=None,n_outer=100,n_inner=100,seed=None):
    """
    Andersen–Broadie dual upper bound 
    """

    # inputs
    S0     = np.asarray(sim_params['S0'])
    d      = S0.size
    r, q   = sim_params['r'],sim_params['q']
    T, vol = sim_params['T'],sim_params['vol']
    N      = sim_params['time_steps']
    dt     = T/N
    disc   = np.exp(-r*dt)
    K     = opt_params['strike']
    style = opt_params.get('style','put')

    disc_pow = np.exp(-r*np.arange(N+1)*dt)

    if model == 'Heston':
        v0      = heston_params['v0']
        kappa   = heston_params['kappa']
        theta   = heston_params['theta']
        sigma_h = heston_params['sigma']

    def phi(S, V=None):
        if reg_params['method'] in ('RLSM','DRLSM'): # The hidden feature space
            A, act = reg_params['A'],reg_params['act']
            phi = np.hstack([act(S@A),np.ones((S.shape[0],1))])
            if reg_params.get('include_payoff',False):
                phi = np.hstack([phi,payoff(S,K,style)[:,None]])
            if model == 'Heston':
                phi = np.hstack([phi,act(V@A)])
        else:                                 # Classic polynomial basis
            phi   = np.hstack([S**k for k in range(reg_params['num_basis']+1)]) 
            if reg_params.get('include_payoff', False):
                phi = np.hstack([phi, payoff(S,K,style)[:,None]])
            if model == 'Heston':
                phi = np.hstack([phi]+[V**k for k in range(1,reg_params['num_basis']+1)])
        return phi

    def CV(j, S, V=None):
        beta = beta_dict.get(j)
        return np.zeros(S.shape[0]) if beta is None else phi(S,V)@beta

    if seed is not None:
        np.random.seed(seed)

    # Simulation using anthitetic paths
    z1 = np.random.randn((n_outer+1)//2,N,d)
    z1 = np.concatenate([z1,-z1])[:n_outer]

    S_out = np.empty((n_outer,N+1,d))
    V_out = np.empty_like(S_out) if model == 'Heston' else None
    S_out[:,0] = S0
    if model == 'Heston':
        V_out[:,0] = v0

    if model == 'Black-Scholes':
        incr  = (r-q-0.5*vol**2)*dt+vol*np.sqrt(dt)*z1
        S_out[:,1:] = S0*np.exp(np.cumsum(incr,axis=1))
    else: # Heston increments
        z2 = np.random.randn((n_outer+1)//2,N,d)
        z2 = np.concatenate([z2,-z2])[:n_outer]

        for t in range(1, N + 1):
            v_next = np.maximum(V_out[:,t-1],0)
            v_next += kappa*(theta-v_next)*dt+sigma_h*np.sqrt(v_next)*z2[:,t-1]*np.sqrt(dt)
            V_out[:,t] = v_next
            s_prev = S_out[:,t-1]
            S_out[:,t] = s_prev+(r-q)*s_prev*dt+s_prev*np.sqrt(np.maximum(v_next, 0.0))*z1[:,t-1]*np.sqrt(dt)

    # time to calculate the gaps
    
    def compute_gap(p):
        
        if seed is not None:
            np.random.seed(seed+p)
            
        Vk,Ek,vk = np.zeros(N+1),np.zeros(N+1),np.zeros(N+1)
    
        for t in range(1, N):
            S_t = S_out[p,t:t+1]
            V_t = V_out[p,t:t+1] if model == 'Heston' else None
    
            if payoff(S_t,K,style)>=CV(t,S_t,V_t):  
                vk[t] = 1
                Vk[t] = payoff(S_t,K,style)[0]*disc_pow[t]
    
            steps_left = N-t
            Z1 = np.random.randn((n_inner+1)//2,steps_left,d)
            Z1 = np.concatenate([Z1,-Z1])[:n_inner]
    
            if model == 'Heston':
                Z2 = np.random.randn((n_inner+1)//2,steps_left,d)
                Z2 = np.concatenate([Z2,-Z2])[:n_inner]
    
            pay = np.zeros(n_inner)
            alive = np.ones(n_inner,bool)
    
            S_in = np.repeat(S_t,n_inner,axis=0)
            V_in = np.repeat(V_t,n_inner,axis=0) if model=='Heston' else None
            disc_in = disc
    
            for h in range(steps_left):
                if model == 'Black-Scholes':
                    S_in[alive] *= np.exp((r-q-0.5*vol**2)*dt+vol*np.sqrt(dt)*Z1[alive,h])
                else:
                    V_in[alive] = np.maximum(V_in[alive],0)+kappa*(theta-V_in[alive])*dt+sigma_h*np.sqrt(np.maximum(V_in[alive],0))*Z2[alive,h]*np.sqrt(dt)
                    S_in[alive] += (r-q)*S_in[alive]*dt+S_in[alive]*np.sqrt(np.maximum(V_in[alive],0))*Z1[alive,h]*np.sqrt(dt)
    
                ex_now = payoff(S_in[alive],K,style)>=CV(t+h+1,S_in[alive],V_in[alive] if model=='Heston' else None)
    
                if ex_now.any():
                    hit = np.where(alive)[0][np.where(ex_now)[0]]
                    pay[hit] = payoff(S_in[hit],K,style)*disc_in
                    alive[hit] = False
    
                if not alive.any():
                    break
                disc_in *= disc
    
            if alive.any():
                pay[alive] = payoff(S_in[alive],K,style)*disc_in
    
            if vk[t] == 0:
                Vk[t] = pay.mean()*disc_pow[t]
            Ek[t] = pay.mean()*disc_pow[t]
    
        term = payoff(S_out[p,-1:],K,style)[0]*disc_pow[N]
        Vk[-1] = Ek[-1] = term
    
        M = np.zeros(N+1)
        M[1] = Vk[1]
        for k in range(2,N+1):
            M[k] = M[k-1]+(Vk[k]-Vk[k-1])-vk[k-1]*(Ek[k-1]-Vk[k-1])
    
        disc_pay = payoff(S_out[p,1:],K,style)*disc_pow[1:]
        return np.max(disc_pay-M[1:])
    
    # Use all cores 
    gaps = Parallel(n_jobs=-1)(delayed(compute_gap)(p) for p in range(n_outer))

    return np.mean(gaps)


