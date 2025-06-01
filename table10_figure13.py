import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from upper_bound import dual_upper_bound
from price_optimizer import american_price
from CRR import CRR_put


# Initial configuration
sim_params = {
    'S0':       100, 
    'T':        1, 
    'vol':      0.2, 
    'r':        0.02,
    'q':        0.0,
    'num_paths':2**18,
    'time_steps':10
}

opt_params = {
    'strike': 100,
    'style':  'geoput'
}

reg_params = {
    'num_basis':      3, 
    'method':        'RLSM', 
    'include_payoff': True,
    'p':              None # To be optimized for RLSM and DRLSM
}

heston_cfg = {'v0': 0.01,'kappa':2.0,'theta': 0.01,'sigma':0.2,'rho':-0.3} 

L_out = 2**11
L_in = 500

# Setup to be tested
und_list   = [5,10,50,100]
methods    = ['LSM','DLSM','RLSM','DRLSM']

seed = 1

total_combos = len(list(itertools.product(methods,und_list)))


def sim(model):
    
    step = 0

    # Run the test
    records = []
    for method in methods:
        # pick basis size per method
        nb = 10 if method in ('RLSM','DRLSM') else 3
        
        for d in und_list:
    
                # Override paramters
                sp               = sim_params.copy()
                sp['S0']         = [sp['S0']]*d
                rp               = reg_params.copy()
                rp['method']     = method
                rp['num_basis']  = nb

                t0 = time.time()
                in_p,out_p,beta_dict = american_price(sp,opt_params,rp,model,heston_cfg,seed,out_of_sample= True)
                dual_gap = dual_upper_bound(sp,opt_params,beta_dict,rp,model,heston_cfg,L_out,L_in,seed+100)
                elapsed = time.time()-t0
                
                records.append({
                    'Method':       method,
                    'd':            d,
                    'out_p':        out_p,
                    'upper_bound':  out_p+dual_gap,
                    'Run_Time_sec': elapsed
                    
                })
                
                step += 1

                print(f"Step {step} out of {total_combos} completed.")
    
    # The table
    df   = pd.DataFrame(records)
    df.sort_values(by=['Method','d'],inplace=True)
    
    if model == 'Black-Scholes':
        n = 2000
        df['CRR'] = df.apply(lambda x: CRR_put(sim_params['S0'],sim_params['T'],opt_params['strike'],sim_params['r'],
                                               sim_params['vol']/np.sqrt(x['d']),n*sim_params['T'],sim_params['time_steps'],
                                               sim_params['q']+(sim_params['vol']**2-(sim_params['vol']/np.sqrt(x['d']))**2)/2),axis=1)
    else:
        df['CRR'] = np.nan
        
    df['Method'] = pd.Categorical(df['Method'],categories=methods,ordered=True)
    df.sort_values(by=['Method','d'], inplace=True)

    df['Scenario'] = df.groupby(['d'],sort=False).ngroup()+1
    
    if model == 'Black-Scholes':
        df['rel_err_ub_bp']  = 10000*(df['upper_bound']-df['CRR'])/df['CRR']
    else:
        sc_mean_ub = df.groupby('Scenario')['upper_bound'].transform('mean')
        df['rel_err_ub_bp'] = 10000 * (df['upper_bound']-sc_mean_ub)/sc_mean_ub
    
    

    return df

# The table
df   = sim('Black-Scholes')
df_h = sim('Heston')


methods = ['LSM','DLSM','RLSM','DRLSM']


plt.figure(figsize=(10,8))
plt.grid(alpha=0.3)
fontsize = 14

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for m,col in zip(methods,colors):
    sub = df[df['Method']==m]
    # out-of-sample curve
    plt.plot(sub['Scenario'], sub['rel_err_ub_bp'],ms=4,label=f'{m}',color=col)
    # upper bound
    plt.hlines(sub['rel_err_ub_bp'].mean(),1,4,colors=col,linestyles='--',linewidth=1,label=f'{m} mean')

plt.axhline(0,color='grey',lw=1)
plt.xticks(range(1,5),fontsize=fontsize)
plt.xlabel('Scenario',fontsize=fontsize)
plt.ylabel('Relative Error in basis points',fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize='large', ncol=2)
plt.tight_layout()
plt.show()


means = df.groupby('Method')['rel_err_ub_bp'].mean()
print(means)



methods = ['LSM','DLSM','RLSM','DRLSM']


plt.figure(figsize=(10,8))
plt.grid(alpha=0.3)
fontsize = 14

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for m,col in zip(methods,colors):
    sub = df_h[df_h['Method']==m]
    # out-of-sample curve
    plt.plot(sub['Scenario'], sub['rel_err_ub_bp'],ms=4,label=f'{m}',color=col)
    # upper bound
    plt.hlines(sub['rel_err_ub_bp'].mean(),1,4,colors=col,linestyles='--',linewidth=1,label=f'{m} mean')

plt.axhline(0,color='grey',lw=1)
plt.xticks(range(1,5),fontsize=fontsize)
plt.xlabel('Scenario',fontsize=fontsize)
plt.ylabel('Performance Deviation in basis points',fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize='large', ncol=2)
plt.tight_layout()
plt.show()


means = df_h.groupby('Method')['rel_err_ub_bp'].mean()
print(means)

































