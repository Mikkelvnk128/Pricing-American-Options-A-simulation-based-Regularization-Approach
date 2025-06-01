import numpy as np

def simulate_gbm(sim_params):
    '''
    Simulation under the Black-Scholes model using a geometric brownian motion
    '''
    S0         = np.array(sim_params['S0'])
    T          = sim_params['T']
    vol        = sim_params['vol']
    r          = sim_params['r']
    q          = sim_params['q']
    num_paths  = sim_params['num_paths']
    time_steps = sim_params['time_steps']
    dt         = T/time_steps

    # dimension
    try:
        d = len(S0)
    except TypeError:
        d = 1

    # If one dimension only simulate nume_paths one time
    if d == 1:
        S = np.zeros((num_paths,time_steps+1))
        S[:, 0] = S0
        for t in range(1,time_steps+1): # Simulation through all time steps
            Z = np.random.randn((num_paths+1)//2)
            Z = np.concatenate([Z,-Z])[:num_paths] # Antithetic paths
            S[:,t] = S[:,t-1]*np.exp((r-q-0.5*vol**2)*dt+vol*np.sqrt(dt)*Z)
            
    else: # For several underlyings, simulate for one extra dimension d
        S = np.zeros((num_paths,time_steps+1,d))
        S[:,0,:] = S0
        vol,r,q = np.full(d,vol),np.full(d,r),np.full(d,q) # Make as vectors for all d (constants)
        for t in range(1,time_steps+1): # Simulation through all times (for d dimensions)
            Z = np.random.randn((num_paths+1)//2,d)
            Z = np.concatenate([Z,-Z])[:num_paths]
            S[:,t,:] = S[:,t-1,:]*np.exp((r-q-0.5*vol**2)*dt+vol*np.sqrt(dt)*Z)
    return S



def simulate_heston(sim_params,heston_params):
    '''
    Simulate Heston paths via explicit Euler–Maruyama,
    '''
    S0         = np.array(sim_params['S0'])
    T          = sim_params['T']
    r          = sim_params['r']
    q          = sim_params['q']
    num_paths  = sim_params['num_paths']
    time_steps = sim_params['time_steps']
    v0         = heston_params['v0']
    kappa      = heston_params['kappa']
    theta      = heston_params['theta']
    sigma      = heston_params['sigma']
    rho        = heston_params['rho']
    dt         = T/time_steps

    # dimension
    try:
        d = len(S0)
    except TypeError:
        d = 1

    # If one dimension only simulate nume_paths one time
    if d == 1:
        S = np.zeros((num_paths,time_steps+1))
        v = np.zeros((num_paths,time_steps+1))
        S[:,0],v[:,0] = S0,v0  

        for i in range(1, time_steps+1): # Simulation through all time steps
            Z1 = np.random.randn((num_paths+1)//2)
            Z2 = np.random.randn((num_paths+1)//2)
            Z1 = np.concatenate([Z1,-Z1])[:num_paths]# Antithetic paths
            Z2 = np.concatenate([Z2,-Z2])[:num_paths]# Antithetic paths
            dW = Z1*np.sqrt(dt)
            dZ = (rho*Z1+np.sqrt(1-rho**2)*Z2)*np.sqrt(dt)

            v[:,i] = (v[:,i-1]+(-kappa*(np.maximum(v[:,i-1],0)-theta))*dt+sigma*np.sqrt(np.maximum(v[:,i-1],0))*dZ)
            S[:,i] = (S[:,i-1]+(r-q)*S[:,i-1]*dt+S[:,i-1]*np.sqrt(np.maximum(v[:,i],0))*dW)

    else:  # For several underlyings, simulate for one extra dimension d
        S = np.zeros((num_paths,time_steps+1,d))
        v = np.zeros((num_paths,time_steps+1,d))
        S[:, 0, :],v[:, 0, :] = S0,v0

        # Make parameters as vectors for all d (constants)
        r,q,kappa,theta,sigma,rho = np.full(d,r),np.full(d,q),np.full(d,kappa),np.full(d,theta),np.full(d,sigma),np.full(d,rho)

        for i in range(1,time_steps+1):
            Z1 = np.random.randn((num_paths+1)//2,d)
            Z2 = np.random.randn((num_paths+1)//2,d)
            Z1 = np.concatenate([Z1,-Z1])[:num_paths]
            Z2 = np.concatenate([Z2,-Z2])[:num_paths]
            dW = Z1*np.sqrt(dt)
            dZ = (rho[None]*Z1+np.sqrt(1-rho[None]**2)*Z2)*np.sqrt(dt)

            v[:,i,:] = (v[:,i-1,:]+(-kappa[None]*(np.maximum(v[:,i-1,:],0)-theta[None]))*dt+sigma[None]*np.sqrt(np.maximum(v[:,i-1,:],0))*dZ)
            S[:,i,:] = (S[:,i-1,:]+(r[None]-q[None])*S[:,i-1,:]*dt+S[:,i-1,:]*np.sqrt(np.maximum(v[:,i,:],0))*dW)

    return S,v

