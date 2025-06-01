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
    'S0':       36, 
    'T':        1.0, 
    'vol':      0.2, 
    'r':        0.06,
    'q':        0.0,
    'num_paths':2**18,
    'time_steps':50
}

opt_params = {
    'strike': 40,
    'style':  'put'
}

reg_params = {
    'num_basis':      3, 
    'method':        'RLSM', 
    'include_payoff': False,
    'p':              None # To be optimized for RLSM and DRLSM
}

heston_cfg = {'v0': 0.01,'kappa': 2.0,'theta': 0.01,'sigma': 0.2,'rho':   -0.3} 

model = 'Black-Scholes'

L_out = 2**11
L_in = 500

# Setup to be tested
S0_list    = [36,38,40,42,44]
T_list     = [1,2]
vol_list   = [0.2,0.4]
methods    = ['LSM','DLSM','RLSM','DRLSM']

seed = 123

total_combos = len(list(itertools.product(methods,S0_list,T_list,vol_list)))

step = 0 # step counter'

# Run the test
records = []
for method in methods:
    # pick basis size per method
    nb = 10 if method in ('RLSM','DRLSM') else 3

    for S0 in S0_list:
        for T in T_list:
            for vol in vol_list:

                # Override paramters
                sp               = sim_params.copy()
                sp['S0']         = S0
                sp['T']          = T
                sp['vol']        = vol
                rp               = reg_params.copy()
                rp['method']     = method
                rp['num_basis']  = nb

                t0 = time.time()
                
                in_p,out_p,beta_dict = american_price(sp,opt_params,rp,model,heston_cfg,seed,out_of_sample= True)
                dual_gap = dual_upper_bound(sp,opt_params,beta_dict,rp,model,heston_cfg,L_out,L_in,seed+100)
                
                elapsed = time.time()-t0

                records.append({
                    'Method':       method,
                    'S0':           S0,
                    'T':            T,
                    'Vol':          vol,
                    'Price_In':     in_p,
                    'Price_Out':    out_p,
                    'Dual':         dual_gap,
                    'Upper_Bound':  out_p+dual_gap,  
                    'Run_Time_sec': elapsed
                })
                
                step += 1
                
                print(f"Step {step} out of {total_combos} completed.")

# The table
df_b   = pd.DataFrame(records)

# Append the price given by CRR for comparison
n = 2000
df_b['CRR'] = df_b.apply(lambda x: CRR_put(x['S0'],x['T'],opt_params['strike'],sim_params['r'],x['Vol'],n*x['T'],sim_params['time_steps']),axis=1)
df_b['Method'] = pd.Categorical(df_b['Method'],categories=methods,ordered=True)
df_b.sort_values(by=['Method','S0','Vol','T'], inplace=True)

print(df_b)


df_b['Upper_Bound_re']  = 10000*(df_b['Upper_Bound']-df_b['CRR'])/df_b['CRR']

df_b['Scenario'] = df_b.groupby(['S0','T','Vol'],sort=False).ngroup()+1


methods = ['LSM','DLSM','RLSM','DRLSM']


plt.figure(figsize=(10,8))
plt.grid(alpha=0.3)
fontsize = 13

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for m,col in zip(methods,colors):
    sub = df_b[df_b['Method']==m]
    # out-of-sample curve
    plt.plot(sub['Scenario'], sub['Upper_Bound_re'],ms=4,label=f'{m}',color=col)
    # upper bound
    plt.hlines(sub['Upper_Bound_re'].mean(),1,20,colors=col,linestyles='--',linewidth=1,label=f'{m} mean')

plt.axhline(0,color='grey',lw=1)
plt.xticks(range(1,21),fontsize=fontsize)
plt.xlabel('Scenario',fontsize=fontsize)
plt.ylabel('Relative Error in basis points',fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize='large', ncol=2)
plt.tight_layout()
plt.show()


means = df_b.groupby('Method')['Upper_Bound_re'].mean()
print(means)






