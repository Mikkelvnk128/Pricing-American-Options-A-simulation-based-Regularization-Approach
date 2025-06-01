import numpy as np
from scipy.stats import gmean


def payoff(S,K,style='put'):
    '''
    Different types of payoffs, checks the dimension of S and the style. 
    Implemented for 'put', 'call', 'geoput', 'geocall', 'minput', 'maxcall'
    '''
    
    S = np.asarray(S)
    
    if style in ('put','call'):
        S = S.flatten() # Flatten for right interpretation
        
        if style == 'put':
            return np.maximum(K-S,0)
        else:
            return np.maximum(S-K,0)

    if S.ndim > 1: # For higher dimensions we have other types
    
        if style in ('geoput','geocall'):
            P = gmean(S,axis=-1)
            
        elif style == 'minput':
            P = np.min(S,axis=-1)
            
        elif style == 'maxcall':
            P = np.max(S,axis=-1)

        if 'put' in style: # First the 'underlying' style, then the overall style (put/call)
            return np.maximum(K-P,0)
        else:
            return np.maximum(P-K,0)



def payoff_gradient(S,K,style='put'):
    '''
    Returns the gradient of the payoff wrt. to the underlying
    '''
    S = np.asarray(S)+1e-100 # Ensures no dividing by zero
    if S.ndim == 1:
        S = S.reshape(-1,1)
    N, d = S.shape
    grad = np.zeros_like(S)

    if style == 'put':
        mask = (S[:,0]<K)
        grad[mask,0] = -1
        
    elif style == 'call':
        mask = (S[:, 0]>K)
        grad[mask, 0] = 1
        
    elif style in ('geoput', 'geocall'):
        G  = gmean(S,axis=-1,keepdims=True)
        dG = (G/S)/d
        itm = (G<K) if 'put' in style else (G>K)
        sign = -1 if 'put' in style else 1
        grad[itm[:,0],:] = sign*dG[itm[:,0],:]
        
    elif style == 'minput':
        mins   = np.min(S,axis=1,keepdims=True)
        itm    = (mins<K)
        is_min = (S == mins)
        grad[itm[:,0],:] = -1*is_min[itm[:,0],:]
        
    elif style == 'maxcall':
        maxs   = np.max(S,axis=1,keepdims=True)
        itm    = (maxs>K)
        is_max = (S == maxs)
        grad[itm[:,0],:] = is_max[itm[:,0],:]
    else:
        raise ValueError(f'Unknown style {style!r}')

    return grad
