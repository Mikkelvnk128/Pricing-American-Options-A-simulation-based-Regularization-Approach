import numpy as np

# Own imports
from payoff import payoff, payoff_gradient

def fit_regression(reg_params,S_t,cf,strike,discount,style,model='Black-Scholes',
                   v_heston=None,STauStar=None,A=None,activation=None,dactivation=None):
                                                        
    num_basis      = reg_params['num_basis']
    method         = reg_params['method']
    include_payoff = reg_params.get('include_payoff',False) # Fall back to no payoff in regression

    immediate = payoff(S_t,strike,style).reshape(-1,1) # For when payoff is included in regression

    # RLSM method
    if method == 'RLSM':
        
        phi = np.hstack([activation(S_t@A),np.ones((S_t.shape[0],1))])
        
        if include_payoff:
            phi = np.hstack([phi,immediate])
            
        if model == 'Heston':
            phi = np.hstack([phi,activation(v_heston@A)])
            
        beta_hat, *_ = np.linalg.lstsq(phi,cf,rcond=1e-15)

    # LSM method
    elif method == 'LSM':
        
        phi = np.hstack([S_t**i for i in range(num_basis+1)])
        
        if include_payoff:
            phi = np.hstack([phi,immediate])
            
        if model == 'Heston':
            vphi = np.hstack([v_heston**i for i in range(1,num_basis+1)])
            phi  = np.hstack([phi,vphi])
            
        beta_hat, *_ = np.linalg.lstsq(phi,cf,rcond=1e-15)

    # DLSM method
    elif method == 'DLSM':
        
        grad    = payoff_gradient(STauStar,strike,style)
        ZS_full = STauStar/S_t*grad
        div     = np.clip(np.count_nonzero(grad,axis=1),1,None)
        ZS      = np.sum(ZS_full,axis=1)/div         

        phi   = np.hstack([S_t**i for i in range(num_basis+1)])
        dphi  = np.hstack([np.zeros_like(S_t)]+[i*S_t**(i-1) for i in range(1,num_basis+1)])
        dphi *= np.linalg.norm(phi)/np.linalg.norm(dphi)  if S_t.shape[1] != 1 else 1   

        if include_payoff:
            phi  = np.hstack([phi,immediate])
            grad_payoff = np.sum(payoff_gradient(S_t,strike,style),axis=1)/div
            dphi = np.hstack([dphi,grad_payoff[:,None]]) 

        if model == 'Heston':
            v_phi = np.hstack([v_heston**i for i in range(1,num_basis+1)])
            phi   = np.hstack([phi,v_phi])
            dphi  = np.hstack([dphi,np.zeros_like(v_phi)])
          
        if model == 'Black-Scholes':   
            val = (cf**2).sum()
            den = (ZS**2).sum()+1e-100  
        
        else: # Heston
            val  =(cf**2).sum()*(dphi**2).sum()
            den = (ZS**2).sum()*(phi**2).sum()+1e-100 
            
        lam   = val/den 
        lhs = np.vstack([phi,np.sqrt(lam)*dphi])
        rhs = np.concatenate([cf,np.sqrt(lam)*ZS])
        
        beta_hat, *_ = np.linalg.lstsq(lhs,rhs,rcond=1e-15)
        

    # DRLSM method
    elif method == 'DRLSM':
        
        grad    = payoff_gradient(STauStar, strike, style)
        ZS_full = STauStar/S_t*grad
        div     = np.clip(np.count_nonzero(grad,axis=1),1,None)
        ZS      = np.sum(ZS_full,axis=1)/div
        
        phi = np.hstack([activation(S_t@A),np.ones((S_t.shape[0],1))])
        dphi = np.hstack([dactivation(S_t@A)*np.mean(A,axis=0),np.zeros((S_t.shape[0],1))])
        dphi *= np.linalg.norm(phi)/np.linalg.norm(dphi) if S_t.shape[1] != 1 else 1  

        if include_payoff:
            phi  = np.hstack([phi,immediate])
            grad_payoff = np.sum(payoff_gradient(S_t,strike,style),axis=1)/div
            dphi = np.hstack([dphi,grad_payoff[:,None]]) 

        if model == 'Heston':
            v_phi = activation(v_heston@A)
            phi   = np.hstack([phi,v_phi])
            dphi  = np.hstack([dphi,np.zeros_like(v_phi)])

        if model == 'Black-Scholes':   
            val = (cf**2).sum()
            den = (ZS**2).sum()+1e-100 
        
        else: # Heston
            val  =(cf**2).sum()*(dphi**2).sum()
            den = (ZS**2).sum()*(phi**2).sum()+1e-100 
            
        lam   = val/den
        lhs = np.vstack([phi,np.sqrt(lam)*dphi])
        rhs = np.concatenate([cf,np.sqrt(lam)*ZS])
        
        beta_hat, *_ = np.linalg.lstsq(lhs,rhs,rcond=1e-15)

    else:
        raise ValueError(f'Unknown method: {method}')

    return beta_hat, phi