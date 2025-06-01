import numpy as np

def CRR_put(S0,T,K,r,sigma,n,time_steps,div=0):
    dt = T/n
    u  = np.exp(sigma*np.sqrt(dt))
    d  = 1/u
    q  = (np.exp((r-div)*dt)-d)/(u-d)

    step = int(n/(time_steps*T))  # Time steps where we are allowed to exercise (integer approximation)

    j  = np.arange(n+1)
    ST = S0*u**j*d**(n-j)
    V  = np.maximum(K-ST,0)

    for i in range(n-1,-1,-1):
        j    = np.arange(i+1)
        St   = S0*u**j*d**(i-j)
        cont = (q*V[1:]+(1-q)*V[:-1])/np.exp(r*dt)
        
        # Exercise only at time_steps
        if i % step == 0 and (i != 0):
            exer = np.maximum(K-St,0) 
            V    = np.maximum(cont,exer)
        else:
            V    = cont

    return V[0]

